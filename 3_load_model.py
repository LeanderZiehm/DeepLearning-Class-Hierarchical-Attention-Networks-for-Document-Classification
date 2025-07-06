import torch
from main.han_model import HierarchicalAttentionNetwork

# MODEL_SAVE_PATH = "data/new_best_han_model.pth"
# VOCAB_PATH = "data/new_vocabulary2.pkl"

MODEL_SAVE_PATH = "main/files/han_model2025-07-06_01-13-47.pth"
# VOCAB_PATH = "files/vocabulary_2025-07-06_01-13-47.pkl"


def load_trained_model(model_path,device=None):
    """
    Load a trained HAN model from checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print("Checkpoint model_config keys:", checkpoint["model_config"].keys())
    
    #get accepted parameters for HierarchicalAttentionNetwork
    accepted_params = HierarchicalAttentionNetwork.__init__.__code__.co_varnames[1:]
    print("Accepted parameters for HierarchicalAttentionNetwork:", accepted_params)

    model = HierarchicalAttentionNetwork(**checkpoint["model_config"])
    
    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    # Load trained weights
    # model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(f"Best validation accuracy: {checkpoint.get('highest_validation_accuracy', 'no validation accuracy found')}")

    return model, checkpoint["model_config"], checkpoint

model, checkpoint_config, checkpoint = load_trained_model(MODEL_SAVE_PATH)

print(f"Model configuration: {checkpoint_config}")

print(f"Model state dict keys: {list(model.state_dict().keys())[:10]}")  # Print first 10 keys for brevity

print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}")  # Print first 10 keys for brevity

