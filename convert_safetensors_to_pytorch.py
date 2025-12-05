from transformers import AutoModelForCausalLM

# Load model from the safetensors directory
model = AutoModelForCausalLM.from_pretrained("./my_model", 
trust_remote_code=True)

# Save the model in standard PyTorch format (creates pytorch_model.bin)
model.save_pretrained("./my_model", safe_serialization=False)

