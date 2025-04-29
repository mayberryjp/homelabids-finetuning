import json

# Load the LLM Fine-Tuning data
with open('llm_finetune_data_with_ips.json', 'r') as f:
    llm_data = json.load(f)

# Check a few samples
print(llm_data[:2])  # Show first 2 samples for sanity check
