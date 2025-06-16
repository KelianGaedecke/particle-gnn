import torch
from torch_geometric.data import Data

def generate_particles(n=10):
    pos = torch.rand(n, 2)
    x = pos.clone()
    batch = torch.zeros(n, dtype=torch.long)
    return Data(x=x, pos=pos, batch=batch)
