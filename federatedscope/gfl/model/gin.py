import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, BatchNorm1d, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv as GINConv
from torch_geometric.nn import Linear

from federatedscope.core.mlp import MLP
"""
Model param names of GIN:
[
    'convs.0.eps',
    'convs.0.nn.linears.0.weight',
    'convs.0.nn.linears.0.bias',
    'convs.0.nn.linears.1.weight',
    'convs.0.nn.linears.1.bias',
    'convs.0.nn.norms.0.weight',
    'convs.0.nn.norms.0.bias',
    'convs.0.nn.norms.0.running_mean',
    'convs.0.nn.norms.0.running_var',
    'convs.0.nn.norms.0.num_batches_tracked',
    'convs.0.nn.norms.1.weight',
    'convs.0.nn.norms.1.bias',
    'convs.0.nn.norms.1.running_mean',
    'convs.0.nn.norms.1.running_var',
    'convs.0.nn.norms.1.num_batches_tracked',
    'convs.1.eps',
    'convs.1.nn.linears.0.weight',
    'convs.1.nn.linears.0.bias',
    'convs.1.nn.linears.1.weight',
    'convs.1.nn.linears.1.bias',
    'convs.1.nn.norms.0.weight',
    'convs.1.nn.norms.0.bias',
    'convs.1.nn.norms.0.running_mean',
    'convs.1.nn.norms.0.running_var',
    'convs.1.nn.norms.0.num_batches_tracked',
    'convs.1.nn.norms.1.weight',
    'convs.1.nn.norms.1.bias',
    'convs.1.nn.norms.1.running_mean',
    'convs.1.nn.norms.1.running_var',
    'convs.1.nn.norms.1.num_batches_tracked',
]
"""


class GIN_Net(torch.nn.Module):
    r"""Graph Isomorphism Network model from the "How Powerful are Graph
    Neural Networks?" paper, in ICLR'19

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GIN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            first_channel = in_channels if i == 0 else hidden
            second_channel = out_channels if i == max_depth-1 else hidden
            self.convs.append(GINConv(MLP([first_channel, hidden, second_channel], batch_norm=True)))
            
        self.dropout = dropout
        self.jk_linear = Linear(hidden*(max_depth+1), hidden)

    def forward(self, x, edge_index, edge_attr):

        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))            
            xs.append(x)
        x = self.jk_linear(torch.cat(xs, dim=-1))
        return x
