import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, PairNorm, MessagePassing
from torch_geometric.utils import add_self_loops, degree, dropout_edge
from torch_geometric.nn.dense.linear import Linear


class MyGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True):
        super(MyGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.add_self_loops = add_self_loops
        # self.linear = torch.nn.Linear(in_channels, out_channels)
        self.linear = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.linear(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
        # edge_index has shape [2, E]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)  # [N, ]
        deg_inv_sqrt = deg.pow(-0.5)   # [N, ]
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
    

class GCN4NODE(torch.nn.Module):
    def __init__(self, in_channels, n_layers=2, pair_norm=False, activation="relu", add_self_loops=True, drop_edge=0.1):
        super(GCN4NODE, self).__init__()
        self.n_layers = n_layers
        self.pair_norm = pair_norm
        self.add_self_loops = add_self_loops
        self.drop_edge = drop_edge
        self.out_channels = 7

        self.net = nn.ModuleList()
        self.PairNorm = PairNorm()
        self.dropout = nn.Dropout(0.6)
        self.log_softmax = nn.LogSoftmax(dim=1)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        
        hidden_size = 16
        for i in range(1, self.n_layers + 1):
            in_channels = in_channels if i == 1 else hidden_size
            out_channels = self.out_channels if i == self.n_layers else hidden_size
            self.net.append(MyGCNConv(in_channels, out_channels, self.add_self_loops))

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, force_undirected=True)
        for i, conv in enumerate(self.net):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.PairNorm(x)
            if i + 1 < self.n_layers:
                x = self.activation(x)
                x = self.dropout(x)
        x = self.log_softmax(x)
        return x

 
class GCN4LINK(torch.nn.Module):
    def __init__(self, in_channels, n_layers=2, pair_norm=False, activation="relu", add_self_loops=True, drop_edge=0.1):
        super(GCN4LINK, self).__init__()
        self.n_layers = n_layers
        self.pair_norm = pair_norm
        self.add_self_loops = add_self_loops
        self.drop_edge = drop_edge
        self.out_channels = 64

        self.net = nn.ModuleList()
        self.PairNorm = PairNorm()
        self.dropout = nn.Dropout(0.6)
        self.log_softmax = nn.LogSoftmax(dim=1)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        self.conv1 = MyGCNConv(in_channels, 128)
        self.conv2 = MyGCNConv(128, self.out_channels)
        
        hidden_size = 128
        for i in range(1, self.n_layers + 1):
            in_channels = in_channels if i == 1 else hidden_size
            out_channels = self.out_channels if i == self.n_layers else hidden_size
            self.net.append(MyGCNConv(in_channels, out_channels, self.add_self_loops))

    def encode(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, force_undirected=True)
        for i, conv in enumerate(self.net):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = self.PairNorm(x)
            if i + 1 < self.n_layers:
                x = self.activation(x)
                x = self.dropout(x)
        return x
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
