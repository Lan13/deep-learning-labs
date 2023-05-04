import torch
import copy
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from tqdm.auto import tqdm


def train_valid(estimator, train_set, valid_set, lrd):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" %device)

    model = copy.deepcopy(estimator).to(device)
    model.device = device
    # model = torch.nn.DataParallel(copy.deepcopy(estimator), device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lrd, last_epoch=-1)

    n_epochs = 20
    batch_size = 128
    # train_loader = Data.DataLoader(train_set, batch_size=batch_size * len(device_ids), shuffle=True)
    # valid_loader = Data.DataLoader(valid_set, batch_size=batch_size * len(device_ids), shuffle=False)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = Data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    lrs = []

    for epoch in range(1, n_epochs + 1):
        if device == "cuda":
            torch.cuda.empty_cache()
        model.train()
        n_correct = 0
        n_total = 0
        train_loss = []
        for batch in tqdm(train_loader):
            x, true_label = batch
            # preds = model(x.to(device_ids[0]))
            preds = model(x.to(device))
            # loss = criterion(preds, true_label.to(device_ids[0]))
            loss = criterion(preds, true_label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            _, pred_label = torch.max(preds, 1)
            # n_correct += (pred_label == true_label.to(device_ids[0])).sum().item()
            n_correct += (pred_label == true_label.to(device)).sum().item()
            n_total += true_label.shape[0]
        lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

        scheduler.step()
            
        train_acc = n_correct / n_total
        avg_train_loss = sum(train_loss) / len(train_loss)

        model.eval()
        n_correct = 0
        n_total = 0
        valid_loss = []
        for batch in valid_loader:
            x, true_label = batch
            with torch.no_grad():
                # preds = model(x.to(device_ids[0]))
                preds = model(x.to(device))
                # loss = criterion(preds, true_label.to(device_ids[0]))
                loss = criterion(preds, true_label.to(device))
                valid_loss.append(loss.item())
            
            _, pred_label = torch.max(preds, 1)
            # n_correct += (pred_label == true_label.to(device_ids[0])).sum().item()
            n_correct += (pred_label == true_label.to(device)).sum().item()
            n_total += true_label.shape[0]

        valid_acc = n_correct / n_total
        avg_valid_loss = sum(valid_loss) / len(valid_loss)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        print("epoch [%d], train loss: [%.4f], valid loss: [%.4f]" 
            %(epoch, avg_train_loss, avg_valid_loss))
        print("epoch [%d], train accuracy: [%.4f], valid accuracy: [%.4f]" 
            %(epoch, train_acc, valid_acc))
    
    return train_losses, train_accs, valid_losses, valid_accs, lrs


def heatmap(conf_mat, cm_labels, cbarlabel):
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(conf_mat, cmap=plt.cm.Blues)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticks(np.arange(conf_mat.shape[1]))
    ax.set_yticks(np.arange(conf_mat.shape[0]))
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Heatmap")

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            color="white" if conf_mat[i, j] > 500 else "black"
            im.axes.text(j, i, format(conf_mat[i, j], "d"), 
                         horizontalalignment="center", color=color)
    plt.show()

def show_process(train_loss, valid_loss, train_accu, valid_accu, lrs):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

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

    axs[2].plot(lrs)
    axs[2].set_title("Model Learning Rate")
    axs[2].set_xlabel("epoch")
    axs[2].set_ylabel("Learing Rate")
    plt.tight_layout()
    plt.show()
