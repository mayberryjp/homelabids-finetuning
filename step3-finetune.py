import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

# Add your Hugging Face token here
HF_TOKEN = ""  # Replace with your actual token

# Login to Hugging Face Hub
login(token=HF_TOKEN)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
train_file = "./llm_finetune_data_with_ips.json"

# 1. Load dataset
dataset = load_dataset("json", data_files=train_file, split="train")

# 2. Load model with 8-bit quantization
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
    torch_dtype=torch.float16,
    use_flash_attention_2=False
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Define training arguments separately
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama3",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="none",
)

# 3. Trainer config for TRL 0.17.0
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    packing=True,
    dataset_text_field="messages",
    max_seq_length=512  # Add it here instead
)

# 4. Train
trainer.train()

# 5. Save
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
