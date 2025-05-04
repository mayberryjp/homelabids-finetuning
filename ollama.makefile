FROM llama3.2-1b
# or another base model like mistral, phi, etc.

PARAMETER num_ctx 4096
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.1

# (Optional) Define system prompt or metadata
SYSTEM "You are a helpful assistant."

# (Optional) Add a license or documentation
LICENSE "OpenRAIL-M"
