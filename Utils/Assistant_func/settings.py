import torch
import numpy.random as nprand
import random

def set_seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nprand.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def set_device(device):
    return torch.device(device if torch.cuda.is_available() else 'cpu')