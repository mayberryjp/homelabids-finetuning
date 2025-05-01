import json
import os
from tqdm import tqdm

# Dataset directory
dataset_dir = './samples'

# Prepare a list of prompt/response pairs
llm_data = []

# Generate prompt/response from each JSON file
for filename in tqdm(os.listdir(dataset_dir), desc="Processing JSON files"):
    if filename.endswith('.json'):
        filepath = os.path.join(dataset_dir, filename)
        with open(filepath, 'r') as f:
            entry = json.load(f)
            
            # Prepare prompt
            prompt = f"""
            Classify the device with the following details:
            Hostname: {entry['host_info']['dhcp_hostname']}
            Flows: {[f"{flow['destination']}:{flow['port']}" for flow in entry['flows']]}
            Destination IPs: {[flow['destination'] for flow in entry['flows']]} 
            Destination Ports: {[flow['port'] for flow in entry['flows']]}
            DNS Queries: {entry['dns_queries']}
            OS: {entry['host_info']['os_fingerprint']}
            MAC Vendor: {entry['host_info']['mac_vendor']}
            MAC Address: {entry['host_info']['mac_address']}
            DHCP Hostname: {entry['host_info']['dhcp_hostname']}
            Category: {entry['host_info']["icon"]}
            Local Description: {entry['host_info']["local_description"]}
            Classify the device as one of: Smart Thermostat, Smartphone, Laptop, etc.
            """
            
            # Prepare response (the device type)
            response = f"{entry['host_info']["icon"]} {entry['host_info']["local_description"]}"
            
            # Add the prompt/response pair
            llm_data.append({"prompt": prompt.strip(), "completion": response.strip()})

# Save as a JSON file
with open('./llm_finetune_data_with_ips.json', 'w') as f:
    json.dump(llm_data, f, indent=2)

print(f"LLM fine-tune data with destination IPs generated! Saved to ./llm_finetune_data_with_ips.json")
