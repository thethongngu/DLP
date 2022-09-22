import torch

CONDITION_EMBEDDING_DIM = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
