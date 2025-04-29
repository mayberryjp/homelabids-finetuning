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
# Map the dataset to extract the "prompt" as input and "response" as target
def preprocess_function(examples):
    return {"input_text": examples["prompt"], "target_text": examples["response"]}

dataset = load_dataset("json", data_files=train_file, split="train")
dataset = dataset.map(preprocess_function)

# 2. Load model with 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Or float16 if needed
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
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
s

# 4. Configure the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    packing=True,  # Enables efficient packing of sequences
    dataset_text_field="input_text",  # Use the preprocessed "input_text" field
    args={
        "output_dir": "./llama3-finetuned",
        "overwrite_output_dir": True,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 10,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": True,  # or fp16 if necessary
        "report_to": "none",
    }
)

# 5. Train the model
trainer.train()

# 6. Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
