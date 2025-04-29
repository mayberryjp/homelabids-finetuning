import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

# Add your Hugging Face token here
HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token

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

# 3. Define training arguments
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

# 4. Configure the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    packing=True,  # Enables efficient packing of sequences
    dataset_text_field="messages",  # Field in the dataset containing the text
)

# 5. Train the model
trainer.train()

# 6. Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
