import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer

model_name = "deepseek-ai/DeepSeek-R1"  # Change if your model name is different
train_file = "./llm_finetune_data_with_ips.json"  # Path to your generated data

# 1. Load dataset
dataset = load_dataset("json", data_files=train_file, split="train")

# 2. Load model with 4-bit quantization (optional but saves a lot of memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # <-- THIS IS CORRECT
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # Sometimes needed for newer Llama 3 code
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # Safety

# 3. Trainer config
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="messages",  # Special! trl expects list of messages
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,          # Efficiently pack multiple samples together
    args={
        "output_dir": "./fine_tuned_llama3",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "learning_rate": 2e-4,
        "bf16": True,          # If your GPU supports bf16
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "report_to": "none",   # No wandb needed
    }
)

# 4. Train
trainer.train()

# 5. Save
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
