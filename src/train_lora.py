from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
import os

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "lora-llm-finetuned"

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
).cuda()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config).cuda()

dataset = load_dataset("json", data_files="data/fine_tune_data.jsonl", split="train")

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

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    remove_unused_columns=False
)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Pesos LoRA guardados en {OUTPUT_DIR}")
