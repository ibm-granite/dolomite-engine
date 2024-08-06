from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dolomite_engine import hf_models

SYSTEM_PROMPT="<|system|>\nYou are an AI assistant developed by IBM. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
USER_PROMPT="<|user|>\n{value}\n"
ASSISTANT="<|assistant|>\n"


text='def factorial(x):'
prompt = SYSTEM_PROMPT+USER_PROMPT.format(value=text)+ASSISTANT

model_path='/proj/checkpoints/bathen/models/base/lab_dolomite_step29721'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

x = tokenizer([text], return_tensors="pt")
for i in x:
    x[i] = x[i].to(device)

y = model.generate(**x, max_new_tokens=100)
print(tokenizer.batch_decode(y)[0])

