# Convert the Distributed Model to a single GPU model

import torch
from model import Transformer, TransformerConfig

def convert_ddp_to_single_model(checkpoint_path: str, output_path: str):
    """
    Convert a DDP (DistributedDataParallel) model checkpoint to a single model checkpoint.
    
    Args:
        checkpoint_path (str): Path to the DDP model checkpoint
        output_path (str): Path where the converted model will be saved
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create a new state dict without the 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    print("Converting DDP model to single model...")
    for key, value in state_dict.items():
        if key.startswith('module.'):
            # Remove the 'module.' prefix
            new_key = key[7:]  # len('module.') == 7
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Create a new checkpoint dictionary
    new_checkpoint = {
        'model_state_dict': new_state_dict,
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None),
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0),
    }
    
    # Save the converted checkpoint
    print(f"Saving converted model to {output_path}")
    torch.save(new_checkpoint, output_path)
    print("Conversion completed successfully!")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "models/TNMT_Beta.pt"  # Your DDP model path
    output_path = "models/TNMT_Beta_single.pt"  # Where to save the converted model
    
    convert_ddp_to_single_model(checkpoint_path, output_path)