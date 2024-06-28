from transformers import set_seed

from .arguments import UnshardingArgs, get_args
from .checkpointing import load_checkpoint_for_training
from .distributed import wrap_model_for_distributed_training
from .enums import Mode
from .model_wrapper import get_model, log_model
from .utils import init_distributed, setup_tf32


def main() -> None:
    """main program"""

    mode = Mode.training

    setup_tf32()

    args: UnshardingArgs = get_args(mode)

    # initialize distributed with nccl for multi-node communications
    init_distributed(
        tensor_parallel_size=args.distributed_args.tensor_parallel_size,
        data_parallel_size=args.distributed_args.data_parallel_size,
        data_parallel_replication_world_size=args.distributed_args.zero_topology.data_parallel_replication_world_size,
        data_parallel_sharding_world_size=args.distributed_args.zero_topology.data_parallel_sharding_world_size,
        timeout_minutes=args.distributed_args.timeout_minutes,
    )
    set_seed(args.random_args.seed)

    model = get_model(args, mode)
    model, optimizer, lr_scheduler = wrap_model_for_distributed_training(args, model)

    log_model(model)
    load_checkpoint_for_training(args, model, optimizer, lr_scheduler, None)

    model.unshard()

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        print(name, param.shape)


if __name__ == "__main__":
    main()
