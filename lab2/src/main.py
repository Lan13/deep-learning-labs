# import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from funcs import *
from models import *

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    # device_ids = [0, 1]

    # load dataset
    data_root_dir = "/amax/home/junwei/deep-learning/CIFAR10"
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_root_dir, train=True, transform=transformer, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root_dir, train=False, transform=transformer, download=True)
    generator = torch.Generator().manual_seed(1689)
    trainset, validset = Data.random_split(train_dataset, [0.8, 0.2], generator)

    # show dataset
    # row = 5
    # column = 5
    # labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # fig, axes = plt.subplots(row, column, figsize=(8, 8))
    # axes = axes.ravel()

    # for i in range(row * column):
    #     img, label = trainset[i]
    #     axes[i].imshow(img.permute(1,2,0))
    #     axes[i].set_title("True: %s" %(labels[label]))
    #     axes[i].axis('off')
    #    plt.subplots_adjust(wspace=0.5)

    train_losses3, train_accs3, valid_losses3, valid_accs3, lrs3 = train_valid(CNN(kernel_size=3), trainset, validset, lrd=1)
    # show_process(train_losses3, valid_losses3, train_accs3, valid_accs3, lrs3)

    ######## Network Depth
    train_losses2, train_accs2, valid_losses2, valid_accs2, lrs2 = train_valid(CNN2(kernel_size=3), trainset, validset, lrd=1)
    # show_process(train_losses2, valid_losses2, train_accs2, valid_accs2, lrs2)

    train_losses4, train_accs4, valid_losses4, valid_accs4, lrs4 = train_valid(CNN4(kernel_size=3), trainset, validset, lrd=1)
    # show_process(train_losses4, valid_losses4, train_accs4, valid_accs4, lrs4)

    ######## Normalization
    train_lossesN, train_accsN, valid_lossesN, valid_accsN, lrsN = train_valid(CNN_N(kernel_size=3), trainset, validset, lrd=1)
    # show_process(train_lossesN, valid_lossesN, train_accsN, valid_accsN, lrsN)

    ######## Dropout
    train_lossesD, train_accsD, valid_lossesD, valid_accsD, lrsD = train_valid(CNN(kernel_size=3, dropout_rate=0), trainset, validset, lrd=1)
    # show_process(train_lossesD, valid_lossesD, train_accsD, valid_accsD, lrsD)

    train_lossesD25, train_accsD25, valid_lossesD25, valid_accsD25, lrsD25 = train_valid(CNN(kernel_size=3, dropout_rate=0.25), trainset, validset, lrd=1)
    # show_process(train_lossesD25, valid_lossesD25, train_accsD25, valid_accsD25, lrsD25)

    train_lossesD75, train_accsD75, valid_lossesD75, valid_accsD75, lrsD75 = train_valid(CNN(kernel_size=3, dropout_rate=0.75), trainset, validset, lrd=1)
    # show_process(train_lossesD75, valid_lossesD75, train_accsD75, valid_accsD75, lrsD75)

    ######## Learning Rate Decay
    train_losses95, train_accs95, valid_losses95, valid_accs95, lrs95 = train_valid(CNN(kernel_size=3), trainset, validset, lrd=0.95)
    # show_process(train_losses95, valid_losses95, train_accs95, valid_accs95, lrs95)

    train_losses90, train_accs90, valid_losses90, valid_accs90, lrs90 = train_valid(CNN(kernel_size=3), trainset, validset, lrd=0.90)
    # show_process(train_losses90, valid_losses90, train_accs90, valid_accs90, lrs90)

    train_losses80, train_accs80, valid_losses80, valid_accs80, lrs80 = train_valid(CNN(kernel_size=3), trainset, validset, lrd=0.80)
    # show_process(train_losses80, valid_losses80, train_accs80, valid_accs80, lrs80)

    ######## Kernel Size
    train_lossesK5, train_accsK5, valid_lossesK5, valid_accsK5, lrsK5 = train_valid(CNN(kernel_size=5), trainset, validset, lrd=1)
    # show_process(train_lossesK5, valid_lossesK5, train_accsK5, valid_accsK5, lrsK5)

    train_lossesK7, train_accsK7, valid_lossesK7, valid_accsK7, lrsK7 = train_valid(CNN(kernel_size=7), trainset, validset, lrd=1)
    # show_process(train_lossesK7, valid_lossesK7, train_accsK7, valid_accsK7, lrsK7)

    ######## ResNet
    train_lossesR, train_accsR, valid_lossesR, valid_accsR, lrsR = train_valid(ResNet(kernel_size=3), trainset, validset, lrd=0.90)
    # show_process(train_lossesR, valid_lossesR, train_accsR, valid_accsR, lrsR)

    train_lossesR3, train_accsR3, valid_lossesR3, valid_accsR3, lrsR3 = train_valid(ResNet3(kernel_size=3), trainset, validset, lrd=0.9)
    # show_process(train_lossesR3, valid_lossesR3, train_accsR3, valid_accsR3, lrsR3)

    ######## Training On the All Datasets and Testing On the Testing Dataset
    """ ResNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" %device)

    model = ResNet(kernel_size=3).to(device)
    model.device = device
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)

    n_epochs = 20
    batch_size = 128
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        n_correct = 0
        n_total = 0
        train_loss = []
        for batch in tqdm(train_loader):
            x, true_label = batch
            preds = model(x.to(device))
            loss = criterion(preds, true_label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            _, pred_label = torch.max(preds, 1)
            n_correct += (pred_label == true_label.to(device)).sum().item()
            n_total += true_label.shape[0]
        scheduler.step()
            
        train_acc = n_correct / n_total
        avg_train_loss = sum(train_loss) / len(train_loss)

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        print("epoch [%d], train loss: [%.4f], train accuracy: [%.4f]" 
            %(epoch, avg_train_loss, train_acc))
        
    model.eval()
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    n_correct = 0
    n_total = 0
    test_losses = []

    true_labels = torch.Tensor()
    pred_labels = torch.Tensor()

    for batch in tqdm(test_loader):
        x, true_label = batch
        with torch.no_grad():
            preds = model(x.to(device))
            loss = criterion(preds, true_label.to(device))
            test_losses.append(loss.item())
        
        _, pred_label = torch.max(preds, 1)
        n_correct += (pred_label == true_label.to(device)).sum().item()
        n_total += true_label.shape[0]

        true_labels = torch.cat((true_labels, true_label.to("cpu")), 0)
        pred_labels = torch.cat((pred_labels, pred_label.to("cpu")), 0)


    test_acc = n_correct / n_total
    avg_test_loss = sum(test_losses) / len(test_losses)

    print("test loss is: %.4f, test accuracy is: %.4f" %(avg_test_loss, test_acc))"""


    # CNN4, Normalization, Dropout 0.5, Learning Rate Decay 0.90, Kernel Size 3x3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on %s" %device)

    model = CNN4(kernel_size=3).to(device)
    model.device = device
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)

    n_epochs = 20
    batch_size = 128
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        n_correct = 0
        n_total = 0
        train_loss = []
        for batch in tqdm(train_loader):
            x, true_label = batch
            preds = model(x.to(device))
            loss = criterion(preds, true_label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            _, pred_label = torch.max(preds, 1)
            n_correct += (pred_label == true_label.to(device)).sum().item()
            n_total += true_label.shape[0]
        scheduler.step()
            
        train_acc = n_correct / n_total
        avg_train_loss = sum(train_loss) / len(train_loss)

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        print("epoch [%d], train loss: [%.4f], train accuracy: [%.4f]" 
            %(epoch, avg_train_loss, train_acc))
        
    model.eval()
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    n_correct = 0
    n_total = 0
    test_losses = []

    true_labels = torch.Tensor()
    pred_labels = torch.Tensor()

    for batch in tqdm(test_loader):
        x, true_label = batch
        with torch.no_grad():
            preds = model(x.to(device))
            loss = criterion(preds, true_label.to(device))
            test_losses.append(loss.item())
        
        _, pred_label = torch.max(preds, 1)
        n_correct += (pred_label == true_label.to(device)).sum().item()
        n_total += true_label.shape[0]

        true_labels = torch.cat((true_labels, true_label.to("cpu")), 0)
        pred_labels = torch.cat((pred_labels, pred_label.to("cpu")), 0)


    test_acc = n_correct / n_total
    avg_test_loss = sum(test_losses) / len(test_losses)

    print("test loss is: %.4f, test accuracy is: %.4f" %(avg_test_loss, test_acc))

    # confusion matrix
    # labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # cm = confusion_matrix(true_labels.to("cpu"), pred_labels.to("cpu"))
    # heatmap(cm, labels, cbarlabel="count of predictions")

    # row = 5
    # column = 5
    # labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # fig, axes = plt.subplots(row, column, figsize=(10, 10))
    # axes = axes.ravel()

    # for i in range(row * column):
    #     img, label = test_dataset[i]
    #     axes[i].imshow(img.permute(1,2,0))
    #     axes[i].set_title("True: %s\nPred: %s" %(labels[label], labels[int(pred_labels[i])]))
    #     axes[i].axis('off')
    #     plt.subplots_adjust(wspace=0.5)
