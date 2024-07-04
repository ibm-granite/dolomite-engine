import torch
import torch.distributed
from transformers import AutoTokenizer

from dolomite_engine.hf_models import GPTDolomiteForCausalLM_TP
from dolomite_engine.utils import CUDA_RNGStatesTracker, ProcessGroupManager, set_cuda_rng_tracker


# initialize distributed
torch.distributed.init_process_group("nccl")

device_count_per_node = torch.cuda.device_count()
local_rank = torch.distributed.get_rank() % device_count_per_node

torch.cuda.set_device(local_rank)

ProcessGroupManager(tensor_parallel_size=8)

# this is needed when combining different kinds of parallelism for training
# leave as is if unaware of what you are doing
cuda_rng_tracker = CUDA_RNGStatesTracker()
cuda_rng_tracker.add("tensor-parallel-seed", 42)
set_cuda_rng_tracker(cuda_rng_tracker)


model_name = "save/"

model = GPTDolomiteForCausalLM_TP.from_pretrained(model_name, tensor_parallel_word_embeddings=True)

if torch.distributed.get_rank() == 0:
    print(model)

# set model to eval mode
model.eval()

# happy generation
text = "def generate():"
tokenizer = AutoTokenizer.from_pretrained(model_name)

x = tokenizer([text], return_tensors="pt")

for i in x:
    x[i] = x[i].to(torch.cuda.current_device())

y = model.generate(**x, max_new_tokens=100)

if torch.distributed.get_rank() == 0:
    print(tokenizer.batch_decode(y)[0])
