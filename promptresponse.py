import json
import os
from tqdm import tqdm

# Dataset directory
dataset_dir = './samples/'
promptresponse_dir = './promptresponse/'

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
            IP: {entry['ip_address']}
            Hostname: {entry['host_info']['dhcp_hostname']}
            Flows: {[flow['port'] for flow in entry['flows']]}
            DNS Queries: {entry['dns_queries']}
            OS: {entry['host_info']['os_fingerprint']}
            Classify the device as one of: Smart Thermostat, Smartphone, Laptop, etc.
            """
            
            # Prepare response (the device type)
            response = entry['label']
            
            # Add the prompt/response pair
            llm_data.append({"prompt": prompt.strip(), "response": response.strip()})

# Save as a JSON file
with open(f'{promptresponse_dir}/promptllm_finetune_data.json', 'w') as f:
    json.dump(llm_data, f, indent=2)

print(f"LLM fine-tune data generated! Saved to {promptresponse_dir}/llm_finetune_data.json")
