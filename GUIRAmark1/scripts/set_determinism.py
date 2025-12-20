import os
import random
import numpy as np
import torch

def set_determinism(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    set_determinism()
    print("Determinism enabled.")
    # Example of a random operation
    print("PyTorch random tensor:", torch.randn(2, 2))
