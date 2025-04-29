from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset from JSON files (adjust as needed)
dataset_dir = './samples'  # Current directory
data = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            entry = json.load(f)
            
            # Extract relevant fields and convert to features
            flows_ports = ', '.join([str(flow['port']) for flow in entry['flows']])
            destination_ips = ', '.join([flow['destination'] for flow in entry['flows']])
            flows = ', '.join([str(f"{flow['destination']}:{flow['port']}") for flow in entry['flows']]),
            # Create a combined string that includes IP, hostname, OS, flows ports, and destination IPs
            features = {
                "text": f"IP: {entry['ip_address']}, Host: {entry['host_info']['dhcp_hostname']}, OS: {entry['host_info']['os_fingerprint']}, Flows Ports: {flows_ports}, Dest IPs: {destination_ips}, Flows: {flows}, MAC Vendor: {entry['host_info']['mac_vendor']}, MAC Address: {entry['host_info']['mac_address']}",
                "label": entry["label"]
            }
            data.append(features)

# Convert the data to a Dataset
df = Dataset.from_dict(data)

# Encode labels
label_encoder = LabelEncoder()
df = df.map(lambda x: {"label": label_encoder.fit_transform([x["label"]])[0]})

# Train-test split
train_dataset, val_dataset = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer and model
model_name = "llama/3.2:1b"  # Replace with the Llama model path or version you're using
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=len(set(df['label'])))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',            # Where to save model checkpoints
    evaluation_strategy="epoch",       # Evaluation frequency
    learning_rate=2e-5,                # Learning rate for the optimizer
    per_device_train_batch_size=8,     # Batch size for training
    per_device_eval_batch_size=8,      # Batch size for evaluation
    num_train_epochs=3,                # Number of epochs
    weight_decay=0.01,                 # Weight decay
    logging_dir='./logs',              # Directory for storing logs
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=model,                          # Pre-trained Llama model
    args=training_args,                   # Training arguments
    train_dataset=train_dataset,          # Training dataset
    eval_dataset=val_dataset,             # Validation dataset
    tokenizer=tokenizer,                  # Tokenizer
)

# Train the model
trainer.train()

# Save the model and tokenizer after training
model.save_pretrained('./fine_tuned_llama_model')
tokenizer.save_pretrained('./fine_tuned_llama_tokenizer')

print("Fine-tuning complete and model saved!")
