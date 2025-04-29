import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

# Add your Hugging Face token here
HF_TOKEN = ""  # Replace with your actual token

# Login to Hugging Face Hub
login(token=HF_TOKEN)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Changed to Meta-Llama 3
train_file = "./llm_finetune_data_with_ips.json"  # Path to your generated data

# 1. Load dataset
dataset = load_dataset("json", data_files=train_file, split="train")

# 2. Load model with 8-bit quantization (optional but saves a lot of memory)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    use_fp8_qdq=False  # Explicitly disable FP8
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Explicitly use float16
    use_flash_attention_2=False  # Disable flash attention which might use FP8
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
    formatting_func=lambda x: x["messages"],  # Use formatting_func instead of dataset_text_field
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
        "fp16": True,          # Instead of bf16
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
