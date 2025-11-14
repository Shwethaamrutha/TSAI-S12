"""
Convert PyTorch checkpoint to SafeTensors format
This avoids malware scanner false positives
"""
import torch
from safetensors.torch import save_file
import json
from dataclasses import dataclass

# Define GPTConfig to match the training script
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# Load original checkpoint
print("Loading checkpoint...")
# PyTorch 2.6+ requires weights_only=False for custom classes like GPTConfig
# We need to register GPTConfig so pickle can find it
import sys
sys.modules['__main__'].GPTConfig = GPTConfig

checkpoint = torch.load('model_checkpoint_final.pt', map_location='cpu', weights_only=False)

# Extract state dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Handle weight sharing: lm_head.weight and transformer.wte.weight share memory
# We need to clone one of them for SafeTensors (it doesn't support shared tensors)
print("Handling weight sharing...")
if 'lm_head.weight' in state_dict and 'transformer.wte.weight' in state_dict:
    # Check if they're the same object (shared memory)
    if state_dict['lm_head.weight'].data_ptr() == state_dict['transformer.wte.weight'].data_ptr():
        print("  Detected weight sharing between lm_head.weight and transformer.wte.weight")
        print("  Cloning transformer.wte.weight to break sharing for SafeTensors...")
        # Clone the embedding weights so they're separate
        state_dict = dict(state_dict)  # Make a copy of the dict
        state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight'].clone()
        print("  ✅ Weight sharing handled")

# Save as SafeTensors
print("Converting to SafeTensors...")
save_file(state_dict, 'model.safetensors')
print("✅ Saved model.safetensors")

# Save config separately
config = {
    'block_size': 1024,
    'vocab_size': 50257,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("✅ Saved config.json")

# Also save training metadata if available
if 'step' in checkpoint or 'loss' in checkpoint:
    metadata = {
        'step': checkpoint.get('step', None),
        'loss': checkpoint.get('loss', None),
    }
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata.json")

print("\n✅ Conversion complete!")
print("Upload these files to HuggingFace Model Hub:")
print("  - model.safetensors")
print("  - config.json")
print("  - metadata.json (optional)")

