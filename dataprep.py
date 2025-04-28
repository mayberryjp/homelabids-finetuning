import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define your dataset directory (unzipped)
dataset_dir = './samples/'  # <-- change this if needed!

# Load all JSON files
data = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            entry = json.load(f)
            data.append(entry)

print(f"Loaded {len(data)} device samples.")

# Optional: Convert to a DataFrame for easier manipulation
df = pd.DataFrame(data)

# Quick peek
print(df.head())

# Example: Build (X, y) datasets
X = df[['ip_address', 'host_info', 'dns_queries', 'flows']]  # inputs
y = df['label']  # target: device type (Smartphone, Printer, etc.)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
