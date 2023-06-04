import torch
import random
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
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


def read_data(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels


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


def train_valid(estimator, train_set, valid_set):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = copy.deepcopy(estimator).to(device)
	criterion = nn.BCELoss()
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
			predictions = model(text).squeeze(1).double()
			loss = criterion(predictions, label)
			
			rounded_preds = torch.round(predictions)
			correct = (rounded_preds == label).float()
			acc = correct.sum() / len(correct)
			
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			train_acc += acc.item()
		scheduler.step()

		train_losses.append(train_loss / len(train_loader))
		train_accs.append(train_acc / len(train_loader))
		print("epcoh [%d], train loss: [%.4f], train accuracy: [%.4f]" % (epoch, train_loss / len(train_loader), train_acc / len(train_loader)))

		valid_loss = 0
		valid_acc = 0
		model.eval()
		for label, text in val_loader:
			with torch.no_grad():
				predictions = model(text).squeeze(1).double()
				loss = criterion(predictions, label).float()
				
				rounded_preds = torch.round(predictions)
				correct = (rounded_preds == label).float()
				acc = correct.sum() / len(correct)
				
				valid_loss += loss.item()
				valid_acc += acc.item()
		
		valid_losses.append(valid_loss / len(val_loader))
		valid_accs.append(valid_acc / len(val_loader))
		print("epoch [%d], valid loss: [%.4f], valid accuracy: [%.4f]" % (epoch, valid_loss / len(val_loader), valid_acc / len(val_loader)))
	
	return train_losses, train_accs, valid_losses, valid_accs
		