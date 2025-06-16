import torch
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import MessagePassing, radius_graph

class ParticleGNN(MessagePassing):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2):
        super().__init__(aggr='mean')
        self.mlp = Sequential(
            Linear(2 * in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, x, pos, batch):
        edge_index = radius_graph(pos, r=1.0, batch=batch)
        return self.propagate(edge_index=edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=1))



