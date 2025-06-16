import torch
from torch_geometric.data import Data

def generate_particles(n=10):
    pos = torch.rand(n, 2)
    x = pos.clone()
    batch = torch.zeros(n, dtype=torch.long)
    return Data(x=x, pos=pos, batch=batch)

# src/train.py
from src.model import ParticleGNN
from src.data_utils import generate_particles

def main():
    model = ParticleGNN()
    data = generate_particles(20)
    delta = model(data.x, data.pos, data.batch)
    next_pos = data.pos + delta
    print("Next position:\n", next_pos)

if __name__ == '__main__':
    main()