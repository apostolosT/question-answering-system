import torch

params = {
    "HIDDEN_DIM": 128,
    "EMB_DIM": 300,
    "NUM_LAYERS": 3,
    "NUM_DIRECTIONS": 2,
    "DROPOUT": 0.3
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
