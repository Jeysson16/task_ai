from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "ruta/a/tu/modelo-8b"
tokenizer   = AutoTokenizer.from_pretrained(model_name)
model       = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def ask_llm(prompt: str, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)
