import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dataset directory (using current directory)
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
                "flows": ', '.join([str(f"{flow['destination']}:{flow['port']}") for flow in entry['flows']]),
                # New features for destination IPs and Ports
                "flows_ports": ', '.join([str(flow['port']) for flow in entry["flows"]]),
                "destination_ips": ', '.join([flow['destination'] for flow in entry["flows"]]),
                "dhcp_hostname": entry["host_info"]["dhcp_hostname"],
                "mac_vendor": entry["host_info"]["mac_vendor"],
                "mac_address": entry["host_info"]["mac_address"]
            }
            
            # Add features and label
            data.append(features | {"label": entry["label"]})

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Encode the labels (target variable)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Handle categorical features (ip_address, hostname, os_fingerprint, destination_ips, flows_ports)
df['ip_address'] = df['ip_address'].astype(str)  # Treat IP address as string
df['hostname'] = df['hostname'].astype(str)      # Treat hostname as string
df['os_fingerprint'] = df['os_fingerprint'].astype(str)  # Treat OS fingerprint as string
df['destination_ips'] = df['destination_ips'].astype(str)  # Treat destination_ips as string
df['flows_ports'] = df['flows_ports'].astype(str)  # Treat ports as string
df['flows'] = df['flows'].astype(str)  # Treat flows as string
df['dhcp_hostname'] = df['dhcp_hostname'].astype(str)  # Treat DHCP hostname as string
df['mac_vendor'] = df['mac_vendor'].astype(str)  # Treat MAC vendor as string
df['mac_address'] = df['mac_address'].astype(str)  # Treat MAC address as string

# Convert categorical columns to numeric using OneHotEncoding or Label Encoding
df_encoded = pd.get_dummies(df, columns=["ip_address", "hostname", "os_fingerprint", "destination_ips", "flows_ports", "flows", "dhcp_hostname", "mac_vendor", "mac_address"])

# Features (X) and Target (y)
X = df_encoded.drop(columns=["label"])
y = df_encoded["label"]

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (for distance-based models like k-NN, or in case of features on different scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Predict on the validation set
y_pred = classifier.predict(X_val_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Save the model (Optional: if you want to persist the model for later use)
import joblib
joblib.dump(classifier, './device_classifier_with_ips_ports_model.pkl')
joblib.dump(scaler, './scaler.pkl')

# Save the training and validation data for further use
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled_df.to_csv('./X_train_scaled_with_ips_ports.csv', index=False)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_val_scaled_df.to_csv('./X_val_scaled_with_ips_ports.csv', index=False)
y_train.to_csv('./y_train_with_ips_ports.csv', index=False)
y_val.to_csv('./y_val_with_ips_ports.csv', index=False)

print("Model and data saved for future use!")
