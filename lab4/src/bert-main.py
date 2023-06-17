import os
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from datas import *
from funcs import *
from models import *
from transformers import BertTokenizer


if __name__ == '__main__':
    device_ids = [0, 1, 2, 3, 4, 5]
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4, 5, 6'
    same_seeds(1689)
    
    # get data
    texts, labels = read_data("aclImdb/train")
    test_texts, test_labels = read_data("aclImdb/test")
    
    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # get dataset
    dataset = CustomTextDataset2(texts, labels, tokenizer)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    # split dataset to train and val
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # default settings
    train_losses, train_accs, valid_losses, valid_accs = train_valid2(
        BERTClassifier(), 
        train_dataset, val_dataset, device_ids, 
        lr=1e-5, n_epochs=5)
    
    # show_process(train_losses, train_accs, valid_losses, valid_accs, "Valid")
    
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()

    # Retrain and test
    model = torch.nn.DataParallel(module=BERTClassifier()).to(device_ids[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=5e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    n_epochs = 5
    batch_size = 64
    # using all training data
    train_set = CustomTextDataset2(texts, labels, tokenizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accs = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device_ids[0])
            attention_mask = batch['attention_mask'].to(device_ids[0])
            labels = batch['labels'].to(device_ids[0])

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            _, predicted = torch.max(preds, dim=1)
            train_acc += torch.sum(predicted == labels).item()
        scheduler.step()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" % (epoch, epoch_train_loss, epoch_train_acc))

    test_dataset = CustomTextDataset2(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    for batch in tqdm(test_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device_ids[0])
            attention_mask = batch['attention_mask'].to(device_ids[0])
            labels = batch['labels'].to(device_ids[0])
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            _, predicted = torch.max(preds, dim=1)
            test_acc += torch.sum(predicted == labels).item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    print("test loss: [%.4f], test accuracy: [%.4f]" % (test_loss, test_acc))

    # test_losses = [test_loss for _ in range(n_epochs)]
    # test_accs = [test_acc for _ in range(n_epochs)]
    # show_process(train_losses, train_accs, test_losses, test_accs, "Test") 