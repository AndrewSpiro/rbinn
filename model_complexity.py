import torch
import os

def check_model_info(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found.")
        return

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print("Detected: State Dictionary (weights only)")
            state_dict = checkpoint
        else:
            print("Detected: Full Model Object")
            state_dict = checkpoint.state_dict()

        layer_names = list(state_dict.keys())
        total_params = sum(p.numel() for p in state_dict.values())
        trainable_params = sum(p.numel() for p in state_dict.values() if p.requires_grad)

        print("-" * 30)
        print(f"Model File: {model_path}")
        print(f"Number of Weight Tensors (Layers): {len(layer_names)}")
        print(f"Total Parameters: {total_params:,}")
        
        print("\nFirst 5 Layer Keys:")
        for name in layer_names[:5]:
            print(f" - {name}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    path = "../VERONA/aspiro_experiment/data/networks/attias_B7A869.pt" 
    check_model_info(path)