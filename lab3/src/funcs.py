import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score


def show_process_node(train_loss, train_accu, valid_loss, valid_accu):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(valid_loss, label="Valid Loss")
    axs[0].set_title("Model Loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].legend(loc='upper left')

    axs[1].plot(train_accu, label="Train Accuracy")
    axs[1].plot(valid_accu, label="Valid Accuracy")
    axs[1].set_title("Model Accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def show_process_link(train_loss, train_auc, valid_loss, valid_auc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(valid_loss, label="Valid Loss")
    axs[0].set_title("Model Loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("Binary Cross Entropy Loss")
    axs[0].legend(loc='upper left')

    axs[1].plot(train_auc, label="Train AUC")
    axs[1].plot(valid_auc, label="Valid AUC")
    axs[1].set_title("Model AUC")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("AUC")
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def train_valid_link(estimator, train_data, val_data, n_epochs=100, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(estimator).to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    train_aucs = []
    valid_losses = []
    valid_aucs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1,)
        edge_label = torch.cat([train_data.edge_label, train_data.edge_label.new_zeros(neg_edge_index.size(1))], dim=0)
        
        pred = model.decode(z, edge_label_index).view(-1)
        train_loss = criterion(pred, edge_label)
        if train_loss.isnan():
            continue
        train_loss.backward()
        optimizer.step()
        pred_label = pred.sigmoid()
        train_auc = roc_auc_score(edge_label.cpu().detach().numpy(), pred_label.detach().cpu().numpy())
        if epoch % 10 == 0 and verbose:
            print("epcoh [%d], train loss: [%.4f], train auc: [%.4f]" %(epoch, train_loss, train_auc))

        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x, val_data.edge_index)
            pred = model.decode(z, val_data.edge_label_index).view(-1)
            val_loss = criterion(pred, val_data.edge_label)
            pred_label = pred.sigmoid()
            val_auc = roc_auc_score(val_data.edge_label.cpu().numpy(), pred_label.cpu().numpy())
        if epoch % 10 == 0 and verbose:
            print("epcoh [%d], valid loss: [%.4f], valid auc: [%.4f]" %(epoch, val_loss, val_auc))

        if epoch == n_epochs and not verbose:
            print("epcoh [%d], train loss: [%.4f], train auc: [%.4f]" %(epoch, train_loss, train_auc))
            print("epcoh [%d], valid loss: [%.4f], valid auc: [%.4f]" %(epoch, val_loss, val_auc))

        train_losses.append(train_loss.item())
        train_aucs.append(train_auc)
        valid_losses.append(val_loss.item())
        valid_aucs.append(val_auc)
    return train_losses, train_aucs, valid_losses, valid_aucs

def train_valid_node(estimator, dataset, n_epochs=100, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = copy.deepcopy(estimator).to(device)
    dataset = dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(dataset.x, dataset.edge_index)
        train_loss = criterion(pred[dataset.train_mask], dataset.y[dataset.train_mask])
        train_loss.backward()
        optimizer.step()
        pred_label = pred.argmax(dim=1)
        train_correct = (pred_label[dataset.train_mask] == dataset.y[dataset.train_mask]).sum()
        train_acc = int(train_correct) / int(dataset.train_mask.sum())
        if epoch % 10 == 0 and verbose:
            print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" %(epoch, train_loss, train_acc))

        model.eval()
        with torch.no_grad():
            pred = model(dataset.x, dataset.edge_index)
            val_loss = criterion(pred[dataset.val_mask], dataset.y[dataset.val_mask])
            pred_label = pred.argmax(dim=1)
            val_correct = (pred_label[dataset.val_mask] == dataset.y[dataset.val_mask]).sum()
            val_acc = int(val_correct) / int(dataset.val_mask.sum())
        if epoch % 10 == 0 and verbose:
            print("epcoh [%d], valid loss: [%.4f], valid accuracy: [%.4f]" %(epoch, val_loss, val_acc))
        
        if epoch == n_epochs and not verbose:
            print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" %(epoch, train_loss, train_acc))
            print("epcoh [%d], valid loss: [%.4f], valid accuracy: [%.4f]" %(epoch, val_loss, val_acc))
        
        train_losses.append(train_loss.item())
        train_accs.append(train_acc)
        valid_losses.append(val_loss.item())
        valid_accs.append(val_acc)
    return train_losses, train_accs, valid_losses, valid_accs