import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Add your Hugging Face token here
HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token

# Login to Hugging Face Hub
login(token=HF_TOKEN)

model_name = "Meta-Llama/Meta-Llama-3-8B"
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

# 3. Define SFTConfig
sft_config = SFTConfig(
    output_dir="./llama3-finetuned",
)

# 4. Configure the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

# 5. Train the model
trainer.train()

# 6. Save the fine-tuned model and tokenizer
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_llama3")
