from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch
import os

# IDs y directorios
BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR = "lora-llm-finetuned"

# Carga tokenizer y modelo en fp16 sin device_map para evitar meta device
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    use_fast=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda()  # Asegura que el modelo esté completamente en la GPU

# Aplica LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config).cuda()  # También pasa el modelo LoRA a GPU

# Dataset
dataset = load_dataset("json", data_files="fine_tune_data.jsonl", split="train")

# Tokenización
def tokenize_batch(examples):
    tokens = tokenizer(
        examples["content"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_batch, batched=True)

# Entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Pesos LoRA guardados en {OUTPUT_DIR}")
