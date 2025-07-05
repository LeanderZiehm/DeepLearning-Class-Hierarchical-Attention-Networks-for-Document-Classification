def log_message(message):
    print(f"[LOG] {message}")

def save_config(config, filepath):
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f)

def load_config(filepath):
    import json
    with open(filepath, 'r') as f:
        return json.load(f)

def set_random_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)