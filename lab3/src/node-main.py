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
    ])
    dataset = Planetoid(root='CORA/', name='Cora', transform=transform)

    # default hyperparameters
    in_channels = dataset[0].num_node_features
    train_losses, train_accs, valid_losses, valid_accs = train_valid_node(
        GCN4NODE(in_channels=in_channels, n_layers=2, pair_norm=False, activation="relu", add_self_loops=True, drop_edge=0.1), 
        dataset[0], n_epochs=200, verbose=True)
    
    # plot train/valid loss/acc
    # show_process_node(train_losses, train_accs, valid_losses, valid_accs)

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
    best_acc = 0

    in_channels = dataset[0].num_node_features
    for loop in loops:
        for layer in layers:
            for drop_edge in drop_edges:
                for pair_norm in pair_norms:
                    for activation in activations:
                        print("loop: {}, layer: {}, drop_edge: {}, pair_norm: {}, activation: {}".format(loop, layer, drop_edge, pair_norm, activation))
                        train_losses, train_accs, valid_losses, valid_accs = train_valid_node(
                            GCN4NODE(in_channels=in_channels, n_layers= layer, 
                                pair_norm=pair_norm, activation=activation, 
                                add_self_loops=loop, drop_edge=drop_edge), 
                            dataset[0], n_epochs=200, verbose=False)
                        acc = sum(valid_accs[-50:]) / 50
                        if acc > best_acc:
                            best_acc =  acc
                            best_loop = loop
                            best_layer = layer
                            best_drop_edge = drop_edge
                            best_pair_norm = pair_norm
                            best_activation = activation
    # print best hyperparameters
    print("acc: {}, loop: {}, layer: {}, drop_edge: {}, pair_norm: {}, activation: {}"
      .format(best_acc, best_loop, best_layer, best_drop_edge, best_pair_norm, best_activation))
    
    # train with best hyperparameters and test on test_set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN4NODE(in_channels=in_channels, n_layers= best_layer, 
                pair_norm=best_pair_norm, activation=best_activation, 
                add_self_loops=best_loop, drop_edge=best_drop_edge).to(device)
    dataset1 = dataset[0].to(device)
    n_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accs = []
    mask = dataset1.train_mask | dataset1.val_mask

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(dataset1.x, dataset1.edge_index)
        train_loss = criterion(pred[mask], dataset1.y[mask])
        train_loss.backward()
        optimizer.step()
        pred_label = pred.argmax(dim=1)
        train_correct = (pred_label[mask] == dataset1.y[mask]).sum()
        train_acc = int(train_correct) / int(mask.sum())
        if epoch % 10 == 0:
            print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" %(epoch, train_loss, train_acc))

        train_losses.append(train_loss.item())
        train_accs.append(train_acc)

    model.eval()
    with torch.no_grad():
        pred = model(dataset1.x, dataset1.edge_index)
        test_loss = criterion(pred[dataset1.test_mask], dataset1.y[dataset1.test_mask]).item()
        pred_label = pred.argmax(dim=1)
        test_correct = (pred_label[dataset1.test_mask] == dataset1.y[dataset1.test_mask]).sum()
        test_acc = int(test_correct) / int(dataset1.test_mask.sum())
        print("\ntest loss: [%.4f], test accuracy: [%.4f]" %(test_loss, test_acc))
    
    # plot train loss/acc and test loss/acc
    test_losses = [test_loss for _ in range(n_epochs)]
    test_accs = [test_acc for _ in range(n_epochs)]
    # This function is shown by "valid", you should modify it to show "test"
    # show_process_node(train_losses, train_accs, test_losses, test_accs)