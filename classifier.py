import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset directory
dataset_dir = './samples'

# List for storing device data
data = []

# Loop through JSON files to extract data for classifier
for filename in os.listdir(dataset_dir):
    if filename.endswith('.json'):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            entry = json.load(f)
            
            # Extract features (X) and label (y)
            features = {
                "ip_address": entry["ip_address"],
                "hostname": entry["host_info"]["dhcp_hostname"],
                "os_fingerprint": entry["host_info"]["os_fingerprint"],
                "num_dns_queries": len(entry["dns_queries"]),
                "num_flows": len(entry["flows"]),
                "flows_ports": ', '.join([str(flow['port']) for flow in entry["flows"]]),
            }
            
            # Add features and label
            data.append(features | {"label": entry["label"]})

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into train and validation sets
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to CSV (or other format) for easy use
X_train.to_csv("/mnt/data/X_train.csv", index=False)
y_train.to_csv("/mnt/data/y_train.csv", index=False)
X_val.to_csv("/mnt/data/X_val.csv", index=False)
y_val.to_csv("/mnt/data/y_val.csv", index=False)

print("Data saved for classifier training!")
