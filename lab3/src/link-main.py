import os 
import copy
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, PairNorm, MessagePassing
from torch_geometric.utils import add_self_loops, degree, dropout_edge

from funcs import *
from models import *

if __name__ == '__main__':
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Load dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device), 
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(root='CORA/', name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]

    # default hyperparameters
    train_losses, train_aucs, valid_losses, valid_aucs = train_valid_link(
        GCN4LINK(in_channels=dataset.num_features, n_layers=2, pair_norm=False, activation="relu", add_self_loops=True, drop_edge=0.1),
        train_data, val_data, n_epochs=200, verbose=True)

    # plot train/valid loss/auc
    # show_process_link(train_losses, train_aucs, valid_losses, valid_aucs)

    # tune hyperparameters
    loops = [True, False]
    layers = [1, 2, 3, 5, 10]
    drop_edges = [0, 0.1, 0.2, 0.5, 0.8]
    pair_norms = [True, False]
    activations = ["relu", "leaky_relu", "tanh", "sigmoid"]

    best_loop = True
    best_layer = 2
    best_drop_edge = 0.2
    best_pair_norm = False
    best_activation = "tanh"
    best_auc = 0

    for loop in loops:
        for layer in layers:
            for drop_edge in drop_edges:
                for pair_norm in pair_norms:
                    for activation in activations:
                        print("loop: {}, layer: {}, drop_edge: {}, pair_norm: {}, activation: {}".format(loop, layer, drop_edge, pair_norm, activation))
                        train_losses, train_aucs, valid_losses, valid_aucs = train_valid_link(
                            GCN4LINK(in_channels=dataset.num_features, n_layers= layer, 
                                pair_norm=pair_norm, activation=activation, 
                                add_self_loops=loop, drop_edge=drop_edge),
                            train_data, val_data, n_epochs=100, verbose=False)
                        auc = sum(valid_aucs[-50:]) / 50
                        if auc > best_auc:
                            best_auc =  auc
                            best_loop = loop
                            best_layer = layer
                            best_drop_edge = drop_edge
                            best_pair_norm = pair_norm
                            best_activation = activation
    # print best hyperparameters
    print("auc: {}, loop: {}, layer: {}, drop_edge: {}, pair_norm: {}, activation: {}"
      .format(best_auc, best_loop, best_layer, best_drop_edge, best_pair_norm, best_activation))
    
    # train with best hyperparameters and test on test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN4LINK(in_channels=dataset.num_features, n_layers=best_layer, pair_norm=best_pair_norm, 
                activation=best_activation, add_self_loops=best_loop, drop_edge=best_drop_edge).to(device)
    train_data1 = train_data.to(device)
    test_data1 = test_data.to(device)
    n_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_losses = []
    train_aucs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data1.x, train_data1.edge_index)
        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(edge_index=train_data1.edge_index, num_nodes=train_data1.num_nodes,
            num_neg_samples=train_data1.edge_label_index.size(1), method='sparse')
        edge_label_index = torch.cat([train_data1.edge_label_index, neg_edge_index], dim=-1,)
        edge_label = torch.cat([train_data1.edge_label, train_data1.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
        pred = model.decode(z, edge_label_index).view(-1)
        train_loss = criterion(pred, edge_label)
        train_loss.backward()
        optimizer.step()
        pred_label = pred.sigmoid()
        train_auc = roc_auc_score(edge_label.cpu().detach().numpy(), pred_label.detach().cpu().numpy())
        if epoch % 10 == 0:
            print("epcoh [%d], train loss: [%.4f], train auc: [%.4f]" %(epoch, train_loss, train_auc))

        train_losses.append(train_loss.item())
        train_aucs.append(train_auc)

    model.eval()
    with torch.no_grad():
        z = model.encode(test_data1.x, test_data1.edge_index)
        pred = model.decode(z, test_data1.edge_label_index).view(-1)
        test_loss = criterion(pred, test_data1.edge_label).item()
        pred_label = pred.sigmoid()
        test_auc = roc_auc_score(test_data1.edge_label.cpu().numpy(), pred_label.cpu().numpy())
        print("\ntest loss: [%.4f], test auc: [%.4f]" %(test_loss, test_auc))
    
    # plot train/test loss/auc
    test_losses = [test_loss for _ in range(n_epochs)]
    test_aucs = [test_auc for _ in range(n_epochs)]
    # This function is shown by "valid", you should modify it to show "test"
    # show_process_link(train_losses, train_aucs, test_losses, test_aucs) 