import model as m
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


# Define the model repository and filename on the Hugging Face Hub
repo_id = "basavyr/adnet"  # Replace with your actual repo
filename = "model.safetensors"  # Replace with the actual safetensors file name

# Download the safetensors file from the hub
safetensor_file_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the safetensors weights into the model
state_dict = load_file(safetensor_file_path)

model = m.Adnet(m.AdnetConfig())

model.load_state_dict(state_dict)

test_tensor = torch.tensor(
    [[60, 9]], dtype=torch.float)
output = model(test_tensor)
print(output)
