import argparse
import os

import torch
import torch.distributed

from dolomite_engine.arguments import TrainingArgs
from dolomite_engine.checkpointing import save_checkpoint
from dolomite_engine.distributed import wrap_model_for_distributed_training
from dolomite_engine.enums import Mode
from dolomite_engine.hf_models import GPTDolomiteForCausalLM_TP
from dolomite_engine.hf_models.models.gpt_dolomite_TP import fix_unsharded_state_dict
from dolomite_engine.model_wrapper import get_model
from dolomite_engine.utils import ProcessGroupManager, load_yaml


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


args = TrainingArgs(**load_yaml(args.config))


tp_world_size = args.distributed_args.tensor_parallel_size
dp_world_size = int(os.getenv("WORLD_SIZE")) // tp_world_size

ProcessGroupManager(tensor_parallel_size=tp_world_size, data_parallel_size=dp_world_size)

tp_rank = ProcessGroupManager.get_tensor_parallel_rank()

if tp_rank == 0:
    with (
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
    ):
        model = get_model(args, Mode.training)
        model.save_pretrained(os.path.join(args.tmp_path, "single_rank"))

torch.distributed.barrier()

# modify args to load the saved single_rank checkpoint
args.model_args.pretrained_config = None
args.model_args.model_name = os.path.join(args.tmp_path, "single_rank")

model_tp = get_model(args, Mode.training)
model_tp, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model_tp)

save_checkpoint(
    args,
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_dataloader=None,
    experiments_tracker=None,
    iteration=0,
    metadata=None,
)

torch.distributed.barrier()

if tp_rank == 0:
    original_state_dict = model.state_dict()

    assert tp_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(tp_state_dict[key])
