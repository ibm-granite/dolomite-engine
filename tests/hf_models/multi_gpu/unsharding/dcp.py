import argparse
import os
import subprocess

import torch
import torch.distributed

from dolomite_engine.arguments import TrainingArgs
from dolomite_engine.checkpointing import save_checkpoint
from dolomite_engine.distributed import wrap_model_for_distributed_training
from dolomite_engine.enums import Mode
from dolomite_engine.model_wrapper import get_model
from dolomite_engine.utils import ProcessGroupManager, load_yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


config = TrainingArgs(**load_yaml(args.config))


tp_world_size = config.distributed_args.tensor_parallel_size
dp_world_size = int(os.getenv("WORLD_SIZE")) // tp_world_size

ProcessGroupManager(tensor_parallel_size=tp_world_size, data_parallel_size=dp_world_size)

tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
global_rank = ProcessGroupManager.get_global_rank()

if global_rank == 0:
    with (
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
    ):
        model = get_model(config, Mode.training)
        model.save_pretrained(os.path.join(args.tmp_path, "single_rank"))

    print(model)

torch.distributed.barrier()

# modify args to load the saved single_rank checkpoint
config.model_args.pretrained_config = None
config.model_args.model_name = os.path.join(args.tmp_path, "single_rank")
config.save_args.save_path = os.path.join(args.tmp_path, "save")

model_tp = get_model(config, Mode.training)
model_tp, optimizer, lr_scheduler = wrap_model_for_distributed_training(config, model_tp)

if global_rank == 0:
    print(model_tp)

save_checkpoint(
    config,
    model=model_tp,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_dataloader=None,
    experiments_tracker=None,
    iteration=0,
    metadata=None,
)

torch.distributed.barrier()


if global_rank == 0:
    consolidated_path = os.path.join(args.tmp_path, "model.pt")

    command = [
        "python",
        "-m",
        "torch.distributed.checkpoint.format_utils",
        "dcp_to_torch",
        os.path.join(config.save_args.save_path, "global_step0", "model"),
        consolidated_path,
    ]
    subprocess.run(command, check=True)

    consolidated_state_dict = torch.load(consolidated_path, "cpu")
    original_state_dict = model.state_dict()

    assert consolidated_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(consolidated_state_dict[key])
