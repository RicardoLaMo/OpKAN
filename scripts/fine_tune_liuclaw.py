# scripts/fine_tune_liuclaw.py
# Template for LoRA fine-tuning LiuClaw agent on H200.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json

def fine_tune_liuclaw(model_id="Qwen/Qwen2.5-32B-Instruct", data_path="data/lora_training_data.jsonl"):
    print(f"🚀 Initializing LoRA Fine-Tuning for {model_id} on H200...")
    
    # 1. Load Data
    # Convert .jsonl to HuggingFace format
    # Each entry: {"instruction": "Analyze context...", "input": "{kan_state}", "output": "{decision_json}"}
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # 2. Load Model in 4-bit/8-bit for efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 4. Training Arguments
    args = TrainingArguments(
        output_dir="models/liuclaw-lora-adapter",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=500,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        report_to="none"
    )
    
    # 5. Trainer (using standard transformers Trainer)
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("🔥 Starting Training...")
    trainer.train()
    
    # 6. Save Adapter
    model.save_pretrained("models/liuclaw-lora-final")
    print("✅ LoRA Fine-Tuning Complete. Adapter saved.")

if __name__ == "__main__":
    # This is a template. Requires real data in data/lora_training_data.jsonl
    # fine_tune_liuclaw()
    pass
