import sklearn.preprocessing._label as _l

import torch

# from torch.serialization import add_safe_globals

from main.han_model import HierarchicalAttentionNetwork

# CONTROL CONSTANTS
ENABLE_TRAINING = True
ENABLE_VALIDATION = True
ENABLE_KFOLD_CV = True
ENABLE_BOOTSTRAP_VALIDATION = True
ENABLE_LEARNING_CURVE = False
SAVE_PLOTS = True


# CONFIGURATION CONSTANTS

MODEL_SAVE_PATH = "main/files/best_han_model.pth"
VOCAB_PATH = "main/files/vocabulary.pkl"




# if you don't want to train the model from scratch we have run the notebook and saved it to google drive: https://drive.google.com/file/d/16HCaqszdyeXFOCiWHkKZ0ljFw0yzu4Hg/view?usp=sharing
# so you can just use load_trained_model after you download the model and put it in this folder :)

def load_trained_model(model_path,device=None):
    """
    Load a trained HAN model from checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    # add_safe_globals([_l.LabelEncoder])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)


    print("Checkpoint model_config keys:", checkpoint["model_config"].keys())
    
    #get accepted parameters for HierarchicalAttentionNetwork
    accepted_params = HierarchicalAttentionNetwork.__init__.__code__.co_varnames[1:]
    print("Accepted parameters for HierarchicalAttentionNetwork:", accepted_params)
    

    KEY_RENAME_MAP = {
        "vocab_size": "vocabulary_size",
        "embed_dim": "embedding_dimmentions",
        "word_gru_hidden": "word_gru_hidden_units",
        "word_gru_layers": "word_gru_layers",
        "word_att_dim": "word_attention_dimmentions",
        "sent_gru_hidden": "sentence_gru_hidden_units",
        "sent_gru_layers": "sentence_gru_layers",
        "sent_att_dim": "sentence_attention_dimmention",
        "num_classes": "number_of_classes"
    }

    accepted_params = HierarchicalAttentionNetwork.__init__.__code__.co_varnames[1:]

    remapped_config = {
        KEY_RENAME_MAP.get(k, k): v for k, v in checkpoint["model_config"].items()
        if KEY_RENAME_MAP.get(k, k) in accepted_params
    }

    if "pretrained_embeddings" in accepted_params and "pretrained_embeddings" not in remapped_config:
        remapped_config["pretrained_embeddings"] = None

    model = HierarchicalAttentionNetwork(**remapped_config)
    
    
    # model = HierarchicalAttentionNetwork(**remapped_config)

    # Remap state dict keys to fix naming mismatches
    def remap_state_dict_keys(state_dict):
        renamed_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("sent_gru", "sentence_gru") \
                     .replace("sent_attention", "sentence_attention") \
                     .replace("sent_context_vector", "sentence_context_vector")
            renamed_dict[new_k] = v
        return renamed_dict

    remapped_state_dict = remap_state_dict_keys(checkpoint["model_state_dict"])

    # Load trained weights
    model.load_state_dict(remapped_state_dict)


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

