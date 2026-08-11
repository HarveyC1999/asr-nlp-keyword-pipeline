## Merge multiple .safetensors files
from safetensors.torch import save_file, load_file
import torch


def combine_safetensors_files(file_paths, output_file):
    combined_tensors = {}

    for file_path in file_paths:
        # Load safetensors file
        tensors = load_file(file_path)

        # Merge tensors
        for key, tensor in tensors.items():
            if key in combined_tensors:
                # Concatenate along dimension 0
                combined_tensors[key] = torch.cat((combined_tensors[key], tensor), dim=0)
            else:
                combined_tensors[key] = tensor

    # Save merged result
    save_file(combined_tensors, output_file)


# List of files to merge
safetensors_files = [
    'model-00001-of-00004.safetensors',
    'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors',
    'model-00004-of-00004.safetensors'
]

# Output safetensors file
combine_safetensors_files(safetensors_files, 'combined_model.safetensors')


## Convert to .bin format
import torch
from safetensors.torch import load_file


# Load safetensors file into PyTorch tensors
def load_safetensors_to_pytorch(file_path):
    return load_file(file_path)


# Save as .bin format
def save_as_bin(tensors, bin_file_path):
    torch.save(tensors, bin_file_path)


safetensors_file = 'combined_model.safetensors'  # Input path
bin_file_path = 'model.bin'  # Output path

tensors = load_safetensors_to_pytorch(safetensors_file)
save_as_bin(tensors, bin_file_path)