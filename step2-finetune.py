import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import os 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add your Hugging Face token here
HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token

# Login to Hugging Face Hub
login(token=HF_TOKEN)

model_name = "meta-llama/Llama-3.2-1B"
train_file = "./llm_finetune_data_with_ips.json"

# 1. Load dataset
# Map the dataset to extract the "prompt" as input and "response" as target
def preprocess_function(examples):
    return {"input_text": examples["prompt"], "target_text": examples["completion"]}

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

# 3. Add LoRA adapters using PEFT
lora_config = LoraConfig(
    r=16,  # Rank of the LoRA update matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # No bias adjustment
    task_type="CAUSAL_LM",  # Task type for causal language modeling
)

model = get_peft_model(model, lora_config)

# 4. Define SFTConfig
sft_config = SFTConfig(
    output_dir="./llama3-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,  # Use bf16 for training
    report_to="none",
    packing=True,  # Enables efficient packing of sequences
    dataset_text_field="input_text",  # Use the preprocessed "input_text" field
)

# 5. Configure the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

# 6. Train the model
trainer.train()

# 7. Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
