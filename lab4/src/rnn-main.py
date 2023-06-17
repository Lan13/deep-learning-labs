import os 

from funcs import *
from datas import *
from models import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, GloVe
from collections import Counter


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(1689)

    # get data
    train_texts, train_labels = read_data("aclImdb/train")
    test_texts, test_labels = read_data("aclImdb/test")

    # get tokenizer
    tokenizer = get_tokenizer('basic_english')
    tokenized_texts = [tokenizer(text) for text in train_texts]
    vectors = GloVe(name='6B', dim=100)

    # Count the frequency of each token in the text data
    counter = Counter(token for text in tokenized_texts for token in text)
    specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = Vocab(counter, vectors=vectors, specials=specials)

    # get dataset
    dataset = CustomTextDataset1(train_texts, train_labels, vocab, tokenizer)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # default settings
    pretrained_embeddings = vocab.vectors
    train_losses, train_accs, valid_losses, valid_accs = train_valid1(
        LSTMClassifier(vocab_size=len(vocab), embedding_dim=100, hidden_dim=64, num_classes=2, 
                    num_layers=2, bidirectional=True, dropout=0.5, 
                    pretrained_embeddings=pretrained_embeddings), 
        train_dataset, val_dataset, lr=1e-3, n_epochs=10)
    
    # show_process(train_losses, train_accs, valid_losses, valid_accs, "Valid")
    
    # best settings and retrain
    lr = 1e-2
    embedding_dim = 300
    hidden_dim = 96 
    num_layers = 3
    bidirectional = True

    # reconstruct dataset for best embedding dim
    tokenizer = get_tokenizer('basic_english')
    tokenized_texts = [tokenizer(text) for text in train_texts]
    vectors = GloVe(name='6B', dim=embedding_dim)

    # Count the frequency of each token in the text data
    counter = Counter(token for text in tokenized_texts for token in text)
    specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = Vocab(counter, vectors=vectors, specials=specials)

    dataset = CustomTextDataset1(train_texts, train_labels, vocab, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_embeddings = vocab.vectors
    model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, 
                        num_classes=2, num_layers=num_layers, bidirectional=bidirectional, 
                        dropout=0.5, pretrained_embeddings=pretrained_embeddings)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

    batch_size = 64
    n_epochs = 25
    # using training set and validation set to train the model
    train_loader = DataLoader(dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accs = []

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

    test_dataset = CustomTextDataset1(test_texts, test_labels, vocab, tokenizer)

    # Create a data loader for the test dataset
    test_loader = DataLoader(test_dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=False)

    test_loss = 0
    test_acc = 0
    model.eval()
    for label, text in tqdm(test_loader):
        with torch.no_grad():
            outputs = model(text)
            loss = criterion(outputs, label.long())
            
            test_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            _, predicted = torch.max(preds, dim=1)
            test_acc += torch.sum(predicted == label).item()
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    print("test loss: [%.4f], test accuracy: [%.4f]" % (test_loss, test_acc))

    # test_losses = [test_loss for _ in range(n_epochs)]
    # test_accs = [test_acc for _ in range(n_epochs)]
    # show_process(train_losses, train_accs, test_losses, test_accs, "Test") 
