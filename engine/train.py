import contextlib
from typing import Union

import torch
from deepspeed import DeepSpeedEngine
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import set_seed

from .arguments import TrainingArgs, get_args
from .checkpointing import load_checkpoint_for_training, save_checkpoint
from .data import get_dataloader, infinite_iterator
from .distributed import wrap_model_for_distributed_training
from .enums import DatasetSplit, DistributedBackend, Mode
from .model import Model
from .utils import (
    ExperimentsTracker,
    ProgressBar,
    RunningMean,
    init_distributed,
    print_rank_0,
    register_profiler,
    register_timer,
    setup_tf32,
)


def track_train_metrics(
    global_step: int,
    train_loss_step: float,
    current_lr: float,
    experiments_tracker: ExperimentsTracker,
    loss_running_mean_tracker: RunningMean,
    progress_bar: ProgressBar,
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        train_loss_step (float): training loss at the current step
        current_lr (float): learning rate at the current step
        experiments_tracker (ExperimentsTracker): metrics tracker
        loss_running_mean_tracker (RunningMean): running mean accumulator for loss
        progress_bar (ProgressBar): progress bar for tracking training progress
    """

    # update loss running mean
    loss_running_mean = loss_running_mean_tracker.add_loss(train_loss_step)

    # track loss
    experiments_tracker.track(
        value=train_loss_step, name="loss", step=global_step, context={"subset": "train", "type": "step"}
    )
    experiments_tracker.track(
        value=loss_running_mean,
        name="loss",
        step=global_step,
        context={"subset": "train", "type": "running_mean"},
    )

    # track learning_rate
    experiments_tracker.track(value=current_lr, name="learning_rate", step=global_step)
    experiments_tracker.info(
        f"step = {global_step}, train_loss (batch) = {train_loss_step}, train_loss (running_mean) = {loss_running_mean}, learning_rate = {current_lr}"
    )

    # update metrics in progress bar
    progress_bar.track(loss_step=train_loss_step, loss_running_mean=loss_running_mean, current_lr=current_lr)


def track_val_metrics(global_step: int, val_loss: float, experiments_tracker: ExperimentsTracker) -> None:
    """tracks metrics like validation loss

    Args:
        global_step (int): global step during training
        val_loss (float): validation loss for the validation data
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    print_rank_0(f"step = {global_step}, val_loss = {val_loss}")
    experiments_tracker.info(f"step = {global_step}, val_loss = {val_loss}")
    experiments_tracker.track(value=val_loss, name="loss", step=global_step, context={"subset": "val"})


@register_profiler("train_step")
@register_timer("train_step")
def train_step(
    model: Union[DeepSpeedEngine, DDP, FSDP],
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    distributed_backend: DistributedBackend,
    train_dataloader: DataLoader,
    gradient_accumulation_steps: int,
) -> float:
    """runs backpropagation and applies the gradient if at the edge of gradient accumulation boundary

    Args:
        model (DeepSpeedEngine, DDP, FSDP): DeepSpeed sharded model
        optimizer (Optimizer): optimizer
        lr_scheduler (LamdaLR): learning rate scheduler
        distributed_backend (DistributedBackend): distributed backend
        train_dataloader (DataLoader): training dataloader
        gradient_accumulation_steps (int): gradient accumulation steps

    Returns:
        float: loss at the current step
    """

    no_sync = model.no_sync if distributed_backend == DistributedBackend.torch else contextlib.nullcontext
    loss = 0
    if distributed_backend == DistributedBackend.torch:
        optimizer.zero_grad()

    with no_sync():
        for _ in range(gradient_accumulation_steps - 1):
            batch = next(train_dataloader)
            loss_micro_step = model(batch)
            loss += loss_micro_step

            # compute gradients
            if distributed_backend == DistributedBackend.deepspeed:
                model.backward(loss_micro_step)
                model.step()
            elif distributed_backend == DistributedBackend.torch:
                loss_micro_step.backward()
            else:
                raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    batch = next(train_dataloader)
    loss_micro_step = model(batch)
    loss += loss_micro_step

    # compute gradients
    if distributed_backend == DistributedBackend.deepspeed:
        model.backward(loss_micro_step)
        model.step()
    elif distributed_backend == DistributedBackend.torch:
        loss_micro_step.backward()
        optimizer.step()
        lr_scheduler.step()
    else:
        raise ValueError(f"unexpected distributed backend ({distributed_backend})")

    loss = loss / gradient_accumulation_steps
    loss = loss.item()

    return loss


def train(
    args: TrainingArgs,
    model: DeepSpeedEngine,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    experiments_tracker: ExperimentsTracker,
) -> None:
    """main training loop for the program

    Args:
        args (TrainingArgs): training args
        model (DeepSpeedEngine): DeepSpeed sharded model
        optimizer (Optimizer): optimizer
        lr_scheduler (LRScheduler): learning rate scheduler
        train_dataloader (DataLoader): training dataloader
        val_dataloader (DataLoader): validation dataloader
        experiments_tracker (ExperimentsTracker): metrics tracker
    """

    num_training_steps = args.training_parameters.num_training_steps
    gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps

    eval_during_training = args.training_parameters.eval_during_training
    eval_interval = args.training_parameters.eval_interval
    distributed_backend = args.distributed_args.distributed_backend
    save_interval = args.save_args.save_interval

    loss_running_mean_tracker = RunningMean()
    progress_bar = ProgressBar(0, num_training_steps)

    model.train()

    train_dataloader = infinite_iterator(train_dataloader)

    # to run on multiple epochs
    for global_step in range(num_training_steps):
        if eval_during_training and global_step % eval_interval == 0:
            val_loss = evaluate(val_dataloader, model)
            track_val_metrics(global_step, val_loss, experiments_tracker)

        if global_step != 0 and global_step % save_interval == 0:
            save_checkpoint(args, model, optimizer, lr_scheduler, global_step)

        loss_step = train_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            distributed_backend=distributed_backend,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        track_train_metrics(
            global_step=global_step,
            train_loss_step=loss_step,
            current_lr=model.lr_scheduler.get_lr()[0]
            if distributed_backend == DistributedBackend.deepspeed
            else lr_scheduler.get_lr()[0],
            experiments_tracker=experiments_tracker,
            loss_running_mean_tracker=loss_running_mean_tracker,
            progress_bar=progress_bar,
        )

        progress_bar.update()

    if eval_during_training:
        val_loss = evaluate(val_dataloader, model)
        track_val_metrics(global_step, val_loss, experiments_tracker)

    if global_step % save_interval != 0:
        save_checkpoint(args, model, optimizer, lr_scheduler, global_step)


@register_profiler("evaluate_dataset")
@torch.no_grad()
def evaluate(val_dataloader: DataLoader, model: Model) -> float:
    """main validation loop for the program

    Args:
        val_dataloader (DataLoader): validation dataloader
        model (DeepSpeedEngine): DeepSpeed sharded model

    Returns:
        float: loss at the current step
    """

    if val_dataloader is None:
        return

    model.eval()

    loss_sum = 0
    micro_step = 0
    progress_bar = ProgressBar(0, len(val_dataloader))

    for batch in val_dataloader:
        loss_value = model(batch).item()
        loss_sum += loss_value
        micro_step += 1
        progress_bar.update()

    loss_mean = loss_sum / micro_step

    model.train()

    return loss_mean


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args: TrainingArgs = get_args(mode)

    # initialize distributed with nccl for multi-node communications
    init_distributed()
    set_seed(args.random_args.seed)

    # setup deepspeed model
    model = Model(args, mode)

    train_dataloader = get_dataloader(
        args,
        split=DatasetSplit.train,
        mode=mode,
        tokenizer=model.tokenizer,
        is_encoder_decoder=model.is_encoder_decoder,
    )

    val_dataloader = None
    if args.training_parameters.eval_during_training:
        val_dataloader = get_dataloader(
            args,
            split=DatasetSplit.val,
            mode=mode,
            tokenizer=model.tokenizer,
            is_encoder_decoder=model.is_encoder_decoder,
        )

    model, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model)

    print_rank_0(model)

    if args.load_args is not None:
        load_checkpoint_for_training(args, model, optimizer, lr_scheduler)

    experiments_tracker = ExperimentsTracker(
        __name__, args.logging_args.experiment_name, args.logging_args.aim_repo, args.logging_args.logdir
    )
    # track all hyperparams in args
    experiments_tracker.log_args(args)

    # main training loop
    train(
        args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        experiments_tracker=experiments_tracker,
    )


if __name__ == "__main__":
    main()
