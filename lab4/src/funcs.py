import torch
import random
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datas import *


def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def show_process(train_loss, train_acc, valid_loss, valid_acc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(train_loss, label="Train Loss")
    axs[0].plot(valid_loss, label="Valid Loss")
    axs[0].set_title("Model Loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("Binary Entropy Loss")
    axs[0].legend(loc='upper left')

    axs[1].plot(train_acc, label="Train Accuracy")
    axs[1].plot(valid_acc, label="Valid Accuracy")
    axs[1].set_title("Model Accuracy")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def train_valid1(estimator, train_set, valid_set):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = copy.deepcopy(estimator).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=5e-3)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

	batch_size = 64
	n_epochs = 10
	train_loader = DataLoader(train_set, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valid_set, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)

	train_losses = []
	train_accs = []
	valid_losses = []
	valid_accs = []

	for epoch in range(1, n_epochs + 1):
		train_loss = 0
		train_acc = 0

		model.train()
		for label, text in tqdm(train_loader):
			optimizer.zero_grad()
			outputs = model(text).squeeze(1).float()
			loss = criterion(outputs, label.long())
			
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			preds = torch.round(torch.sigmoid(outputs))
			_, predicted = torch.max(preds, dim=1)
			train_acc += torch.sum(predicted == label).item()
		scheduler.step()

		epoch_train_loss = train_loss / len(train_loader.dataset)
		epoch_train_acc = train_acc / len(train_loader.dataset)
		train_losses.append(epoch_train_loss)
		train_accs.append(epoch_train_acc)
		print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" % (epoch, epoch_train_loss, epoch_train_acc))

		valid_loss = 0
		valid_acc = 0
		model.eval()
		for label, text in val_loader:
			with torch.no_grad():
				outputs = model(text)
				loss = criterion(outputs, label.long())
				
				valid_loss += loss.item()
				preds = torch.round(torch.sigmoid(outputs))
				_, predicted = torch.max(preds, dim=1)
				valid_acc += torch.sum(predicted == label).item()
		
		epoch_val_loss = valid_loss / len(val_loader.dataset)
		epoch_val_acc = valid_acc / len(val_loader.dataset)
		valid_losses.append(epoch_val_loss)
		valid_accs.append(epoch_val_acc)
		print("epoch [%d], valid loss: [%.4f], valid accuracy: [%.4f]" % (epoch, epoch_val_loss, epoch_val_acc))
	
	return train_losses, train_accs, valid_losses, valid_accs


def train_valid2(estimator, train_set, valid_set, device_ids, lr=5e-5, n_epochs=10):
	# model = copy.deepcopy(estimator).to(device_ids[0])
	model = torch.nn.DataParallel(module=copy.deepcopy(estimator)).to(device_ids[0])
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=lr)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

	batch_size = 64
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

	train_losses = []
	train_accs = []
	valid_losses = []
	valid_accs = []

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

			train_loss += loss.item() * input_ids.size(0)
			preds = torch.round(torch.sigmoid(outputs))
			_, predicted = torch.max(preds, dim=1)
			train_acc += torch.sum(predicted == labels).item()
		scheduler.step()

		epoch_train_loss = train_loss / len(train_loader.dataset)
		epoch_train_acc = train_acc / len(train_loader.dataset)
		train_losses.append(epoch_train_loss)
		train_accs.append(epoch_train_acc)
		print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" % (epoch, epoch_train_loss, epoch_train_acc))

		model.eval()
		val_loss = 0.0
		val_acc = 0.0
		for batch in val_loader:
			with torch.no_grad():
				input_ids = batch['input_ids'].to(device_ids[0])
				attention_mask = batch['attention_mask'].to(device_ids[0])
				labels = batch['labels'].to(device_ids[0])
				outputs = model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
				loss = criterion(outputs, labels)
				val_loss += loss.item() * input_ids.size(0)
				preds = torch.round(torch.sigmoid(outputs))
				_, predicted = torch.max(preds, dim=1)
				val_acc += torch.sum(predicted == labels).item()

		epoch_val_loss = val_loss / len(val_loader.dataset)
		epoch_val_acc = val_acc / len(val_loader.dataset)
		valid_losses.append(epoch_val_loss)
		valid_accs.append(epoch_val_acc)
		print("epoch [%d], valid loss: [%.4f], valid accuracy: [%.4f]" % (epoch, epoch_val_loss, epoch_val_acc))

	return train_losses, train_accs, valid_losses, valid_accs