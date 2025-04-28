from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# 1. Load model and tokenizer
model_name = "NousResearch/Llama-2-7b-chat-hf"  # Or smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
))

# 2. Load and prepare data
with open("device_data.json", "r") as f:
    raw_data = json.load(f)

examples = []
for entry in raw_data:
    ip = entry.get("ip_address", "Unknown IP")
    os = entry.get("host_info", {}).get("os_fingerprint", "Unknown OS")
    hostname = entry.get("host_info", {}).get("lease_hostname", "Unknown Hostname")
    flows = entry.get("flows", [])
    conn_info = ", ".join([f"{f['destination']} (port {f['port']}, proto {f['protocol']})" for f in flows])

    # Build the classification prompt
    prompt = (
        f"Classify the following device:\n\n"
        f"IP Address: {ip}\n"
        f"OS Fingerprint: {os}\n"
        f"Hostname: {hostname}\n"
        f"Connections: {conn_info}\n\n"
        f"Answer with one word only: [Device Type]"
    )

    # Ground truth label: you define it manually or from hostname (example)
    device_type = "Electricity Meter"  # <- your label, depends on dataset

    examples.append({"prompt": prompt, "response": device_type})

# 3. Hugging Face Dataset
dataset = Dataset.from_list(examples)

# 4. Preprocessing
def preprocess_function(examples):
    full_prompt = f"### Instruction:\n{examples['prompt']}\n\n### Response:\n{examples['response']}"
    tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_function)

# 5. Training setup
