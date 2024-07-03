import argparse
import os

import torch
import torch.distributed

from dolomite_engine.arguments import TrainingArgs, UnshardingArgs
from dolomite_engine.checkpointing import load_checkpoint_for_inference, save_checkpoint
from dolomite_engine.distributed import wrap_model_for_distributed_training
from dolomite_engine.enums import Mode
from dolomite_engine.model_wrapper import get_model
from dolomite_engine.utils import ProcessGroupManager, load_yaml


parser = argparse.ArgumentParser()
parser.add_argument("--train-config", type=str)
parser.add_argument("--unshard-config", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


train_config = TrainingArgs(**load_yaml(args.train_config))
unshard_config = UnshardingArgs(**load_yaml(args.unshard_config))

tp_world_size = train_config.distributed_args.tensor_parallel_size
dp_world_size = int(os.getenv("WORLD_SIZE")) // tp_world_size

ProcessGroupManager(tensor_parallel_size=tp_world_size, data_parallel_size=dp_world_size)

tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
global_rank = ProcessGroupManager.get_global_rank()

if global_rank == 0:
    with (
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
    ):
        model = get_model(train_config, Mode.training)
        model.save_pretrained(os.path.join(args.tmp_path, "single_rank"))

torch.distributed.barrier()

# modify args to load the saved single_rank checkpoint
train_config.model_args.pretrained_config = None
train_config.model_args.model_name = os.path.join(args.tmp_path, "single_rank")
train_config.save_args.save_path = os.path.join(args.tmp_path, "save")

iteration = 0
unshard_config.load_args.load_path = train_config.save_args.save_path
unshard_config.load_args.iteration = iteration
unshard_config.unsharded_path = os.path.join(args.tmp_path, "unsharded_path")

model_tp = get_model(train_config, Mode.training)
model_tp, optimizer, lr_scheduler = wrap_model_for_distributed_training(train_config, model_tp)

save_checkpoint(
    train_config,
    model=model_tp,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_dataloader=None,
    experiments_tracker=None,
    iteration=iteration,
    metadata=None,
)

torch.distributed.barrier()


_, _, consolidated_state_dict = load_checkpoint_for_inference(unshard_config, mode=Mode.unsharding, use_meta=False)

if global_rank == 0:
    original_state_dict = model.state_dict()

    assert consolidated_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(consolidated_state_dict[key])

torch.distributed.barrier()
