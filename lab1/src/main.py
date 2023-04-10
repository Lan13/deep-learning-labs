from dnn import *
from func import *


# Info: This Code Will Not Plot Any Figure, Please Read The Report
# For The Display Of The Figures Are Messing
# Because I'm Running On The Google Colab
if __name__ == '__main__':
    # generate dataset
    (X_, y_), (test_X, test_y) = generate_dataset(num=100000)
    # show dataset function curves
    # show_init_func(X_, y_)

    # default settings
    default_train_time, default_valid_loss, default_train_loss = crossvalid4_eval(DNN(), X_, y_)

    # plt.title("Training Loss on Default Parameters")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.plot(default_train_loss)

    # layers tuning
    layers = [2, 4, 5]

    best_layer = 3
    min_layer_loss = default_valid_loss
    layer_train_times = []
    layer_valid_losses = []
    layer_train_losses = []

    for layer in layers:
        print("===== parameter layer: [%d] =====\n" % (layer))
        train_time, valid_loss, layer_train_loss = crossvalid4_eval(DNN(layer=layer), X_, y_, verbose=False)
        layer_train_times.append(train_time)
        layer_valid_losses.append(valid_loss)
        layer_train_losses.append(layer_train_loss)
        if valid_loss < min_layer_loss:
            min_layer_loss = valid_loss
            best_layer = layer

    print("find best layer parameter: [%d]\n" % (best_layer))
    print(layer_train_times)
    print(layer_valid_losses)

    # hidden_sizes tuning
    hidden_sizes = [16, 64, 128]

    best_hidden_size = 32
    min_hidden_size_loss = default_valid_loss
    hidden_size_train_times = []
    hidden_size_valid_losses = []
    hidden_size_train_losses = []

    for hidden_size in hidden_sizes:
        print("===== parameter hidden_size: [%d] =====\n" % (hidden_size))
        train_time, valid_loss, hidden_size_train_loss = crossvalid4_eval(DNN(hidden_size=hidden_size), X_, y_,
                                                                          verbose=False)
        hidden_size_train_times.append(train_time)
        hidden_size_valid_losses.append(valid_loss)
        hidden_size_train_losses.append(hidden_size_train_loss)
        if valid_loss < min_hidden_size_loss:
            min_hidden_size_loss = valid_loss
            best_hidden_size = hidden_size

    print("find best hidden_size parameter: [%d]\n" % (best_hidden_size))
    print(hidden_size_train_times)
    print(hidden_size_valid_losses)

    # activation functions tuning
    activations = ["sigmoid", "tanh"]

    best_activation = "relu"
    min_activation_loss = default_valid_loss
    activation_train_times = []
    activation_valid_losses = []
    activation_train_losses = []

    for activation in activations:
        print("===== parameter activation function: [%s] =====\n" % (activation))
        train_time, valid_loss, activation_train_loss = crossvalid4_eval(DNN(activation=activation), X_, y_,
                                                                         verbose=False)
        activation_train_times.append(train_time)
        activation_valid_losses.append(valid_loss)
        activation_train_losses.append(activation_train_loss)
        if valid_loss < min_activation_loss:
            min_activation_loss = valid_loss
            best_activation = activation

    print("find best activation function parameter: [%s]\n" % (best_activation))
    print(activation_train_times)
    print(activation_valid_losses)

    # learning rates tuning
    lrs = [0.05, 0.01, 0.001]

    best_lr = 0.005
    min_lr_loss = default_valid_loss
    lr_train_times = []
    lr_valid_losses = []
    lr_train_losses = []

    for lr in lrs:
        print("===== parameter learning rate: [%f] =====\n" % (lr))
        train_time, valid_loss, lr_train_loss = crossvalid4_eval(DNN(), X_, y_, lr=lr, verbose=False)
        lr_train_times.append(train_time)
        lr_valid_losses.append(valid_loss)
        lr_train_losses.append(lr_train_loss)
        if valid_loss < min_lr_loss:
            min_lr_loss = valid_loss
            best_lr = lr

    print("find best learning rate parameter: [%f]\n" % (best_lr))
    print(lr_train_times)
    print(lr_valid_losses)

    f, ax = plt.subplots(2, 2)
    f.suptitle('Training Loss')
    plt.tight_layout()

    f.text(0.5, 0, 'epoch', ha='center')
    f.text(0, 0.5, 'loss', va='center', rotation='vertical')

    # Plot Training Loss
    """layer_str = [str(i) for i in layers]
    layer_str.append("3")
    layer_train_losses_copy = copy.deepcopy(layer_train_losses)
    layer_train_losses_copy.append(default_train_loss)
    ax[0][0].set_title('layer')
    for i, label in enumerate(layer_str):
        ax[0][0].plot(layer_train_losses_copy[i], label=label)
    ax[0][0].legend()

    hidden_size_str = [str(i) for i in hidden_sizes]
    hidden_size_str.append("32")
    hidden_size_train_losses_copy = copy.deepcopy(hidden_size_train_losses)
    hidden_size_train_losses_copy.append(default_train_loss)
    ax[0][1].set_title('hidden_size')
    for i, label in enumerate(hidden_size_str):
        ax[0][1].plot(hidden_size_train_losses_copy[i], label=label)
    ax[0][1].legend()

    activation_str = [str(i) for i in activations]
    activation_str.append("relu")
    activation_train_losses_copy = copy.deepcopy(activation_train_losses)
    activation_train_losses_copy.append(default_train_loss)
    ax[1][0].set_title('activation')
    for i, label in enumerate(activation_str):
        ax[1][0].plot(activation_train_losses_copy[i], label=label)
    ax[1][0].legend()

    lr_str = [str(i) for i in lrs]
    lr_str.append("0.005")
    lr_train_losses_copy = copy.deepcopy(lr_train_losses)
    lr_train_losses_copy.append(default_train_loss)
    ax[1][1].set_title('learning rate')
    for i, label in enumerate(lr_str):
        ax[1][1].plot(lr_train_losses_copy[i], label=label)
    ax[1][1].legend()"""

    # Plot Training Time
    """f, ax = plt.subplots(2, 2)
    f.suptitle('Training Time')
    plt.tight_layout()

    layer_str = [str(i) for i in layers]
    layer_str.append("3")
    layer_train_times_copy = copy.deepcopy(layer_train_times)
    layer_train_times_copy.append(default_train_time)
    ax[0][0].set_title('layer')
    ax[0][0].bar(layer_str, layer_train_times_copy, width=0.5)

    hidden_size_str = [str(i) for i in hidden_sizes]
    hidden_size_str.append("32")
    hidden_size_train_times_copy = copy.deepcopy(hidden_size_train_times)
    hidden_size_train_times_copy.append(default_train_time)
    ax[0][1].set_title('hidden_size')
    ax[0][1].bar(hidden_size_str, hidden_size_train_times_copy, width=0.5)

    activation_str = [str(i) for i in activations]
    activation_str.append("relu")
    activation_train_times_copy = copy.deepcopy(activation_train_times)
    activation_train_times_copy.append(default_train_time)
    ax[1][0].set_title('activation')
    ax[1][0].bar(activation_str, activation_train_times_copy, width=0.5)

    lr_str = [str(i) for i in lrs]
    lr_str.append("0.005")
    lr_train_times_copy = copy.deepcopy(lr_train_times)
    lr_train_times_copy.append(default_train_time)
    ax[1][1].set_title('learning rate')
    ax[1][1].bar(lr_str, lr_train_times_copy, width=0.5)"""

    # Plot Validation Loss
    """f, ax = plt.subplots(2, 2)
    f.suptitle('Validation Loss')
    plt.tight_layout()

    layer_str = [str(i) for i in layers]
    layer_str.append("3")
    layer_valid_losses_copy = copy.deepcopy(layer_valid_losses)
    layer_valid_losses_copy.append(default_valid_loss)
    ax[0][0].set_title('layer')
    ax[0][0].bar(layer_str, layer_valid_losses_copy, width=0.5)

    hidden_size_str = [str(i) for i in hidden_sizes]
    hidden_size_str.append("32")
    hidden_size_valid_losses_copy = copy.deepcopy(hidden_size_valid_losses)
    hidden_size_valid_losses_copy.append(default_valid_loss)
    ax[0][1].set_title('hidden_size')
    ax[0][1].bar(hidden_size_str, hidden_size_valid_losses_copy, width=0.5)

    activation_str = [str(i) for i in activations]
    activation_str.append("relu")
    activation_valid_losses_copy = copy.deepcopy(activation_valid_losses)
    activation_valid_losses_copy.append(default_valid_loss)
    ax[1][0].set_title('activation')
    ax[1][0].bar(activation_str, activation_valid_losses_copy, width=0.5)

    lr_str = [str(i) for i in lrs]
    lr_str.append("0.005")
    lr_valid_losses_copy = copy.deepcopy(lr_valid_losses)
    lr_valid_losses_copy.append(default_valid_loss)
    ax[1][1].set_title('learning rate')
    ax[1][1].bar(lr_str, lr_valid_losses_copy, width=0.5)"""

    print("layer: [%d], hidden_size: [%d], activation: [%s], lr: [%f]\n" %
          (best_layer, best_hidden_size, best_activation, best_lr))

    # retrain and validate on test dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # select best parameters
    model = DNN(layer=best_layer, hidden_size=best_hidden_size, activation=best_activation).to(device)
    model.device = device

    criterion = nn.MSELoss()
    # select best learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

    n_epochs = 10
    batch_size = 32

    train_dataset = Data.TensorDataset(X_, y_)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = []
        for batch in tqdm(train_loader):
            x, label = batch
            pred = model(x.to(device))
            loss = criterion(pred, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        print("epoch [%d], train MSE loss: [%f]" % (epoch, train_loss))

    test_set = Data.TensorDataset(test_X, test_y)
    test_loader = Data.DataLoader(test_set, batch_size=1, shuffle=False)

    model.eval()
    valid_losses = []
    preds = []
    for batch in tqdm(test_loader):
        x, true_y = batch
        with torch.no_grad():
            pred = model(x.to(device))
        loss = criterion(pred, true_y.to(device))
        valid_losses.append(loss)
        preds.append(pred)
    valid_loss = sum(valid_losses) / len(valid_losses)
    print("test MSE loss: [%f]\n" % (valid_loss))

    # show_pred_func(test_X, test_y, preds)
