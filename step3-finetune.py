from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============ 1. Load and Prepare Data ============

dataset_dir = './'
data = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            entry = json.load(f)
            flows_ports = ', '.join(str(flow['port']) for flow in entry.get('flows', []))
            destination_ips = ', '.join(flow['destination'] for flow in entry.get('flows', []))
            flows = ', '.join([str(f"{flow['destination']}:{flow['port']}") for flow in entry['flows']])
            features = {
                "text": f"IP: {entry['ip_address']}, Host: {entry['host_info']['dhcp_hostname']}, OS: {entry['host_info']['os_fingerprint']}, Flows Ports: {flows_ports}, Dest IPs: {destination_ips}, Flows: {flows}, MAC Vendor: {entry['host_info']['mac_vendor']}, MAC Address: {entry['host_info']['mac_address']}",
                "label": entry["label"]
            }
            data.append(features)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform([x["label"] for x in data])

# Add encoded labels
for idx, entry in enumerate(data):
    entry["encoded_label"] = labels[idx]

# Split into train/test
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert into HuggingFace Datasets
train_dataset = Dataset.from_dict({
    "text": [x["text"] for x in train_data],
    "label": [x["encoded_label"] for x in train_data],
})

val_dataset = Dataset.from_dict({
    "text": [x["text"] for x in val_data],
    "label": [x["encoded_label"] for x in val_data],
})

# ============ 2. Load Model and Tokenizer ============

model_name = "./llama-3.2-1b"  # local directory or HuggingFace model ID
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# ============ 3. Tokenization ============

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Huggingface expects "labels" field
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ============ 4. Define Training Args ============

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# ============ 5. Trainer Setup ============

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ============ 6. Start Training ============

trainer.train()

# Save model after training
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("✅ Training completed and model saved at ./fine_tuned_llama")
