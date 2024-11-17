import os

from huggingface_hub import login
login(token="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")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/gemma-2-2b-jpn-it"
path = "/workspace/input/llm/gemma-2-2b-jpn-it"
os.makedirs(path,exist_ok=True)

# トークナイザーのダウンロード
tokenizer = AutoTokenizer.from_pretrained(model_name)

# モデルのダウンロード
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer.save_pretrained(path)
model.save_pretrained(path)
