import torch
import copy
import time
import torch.utils.data as Data
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from tqdm.auto import tqdm
from dnn import *


def generate_dataset(num, test_size=0.2, random_state=1689):
    """ 生成数据集，并且划分出训练集和验证集，测试集，默认以8:2的比例划分
    :param num: 总数据集的大小
    :param test_size: 测试集的大小
    :param random_state: 随机种子
    """
    X = torch.empty((num, 1), dtype=torch.float32).uniform_(1, 16)
    y = torch.log2(X) + torch.cos(torch.pi / 2 * X)
    X_, test_X, y_, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_, y_), (test_X, test_y)


def show_init_func(X, y):
    """
    :param X: 函数定义域
    :param y: 函数值域
    """
    plt.title("function curves")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y)
    # plt.show()


def show_pred_func(X, true_y, pred_y):
    """
    :param X: 函数定义域
    :param true_y: 函数真实值域
    :param pred_y: 函数预测值域
    """
    plt.title("function curves")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, torch.tensor(pred_y), label="pred_value")
    plt.scatter(X, true_y, label="true_value")
    plt.legend()
    # plt.show()


def crossvalid4_eval(estimator, X, y, lr=0.005, random_state=1689, verbose=True):
    """ 4折交叉训练并验证
    :param estimator: 需要使用的网络模型
    :param X: 目标数据集
    :param y: 数据集标签
    :param lr: 优化器的学习率
    :param random_state: 随机种子
    :param verbose:
    """

    kf = KFold(n_splits=4, shuffle=True, random_state=random_state)
    avg_train_time = 0
    avg_valid_loss = 0

    for i, (train_index, valid_index) in enumerate(kf.split(X)):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = copy.deepcopy(estimator).to(device)
        model.device = device

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        n_epochs = 10
        batch_size = 32

        print("=== cross validation [%d] ===" % (i + 1))
        train_X, valid_X = X[train_index], X[valid_index]
        train_y, valid_y = y[train_index], y[valid_index]

        train_dataset = Data.TensorDataset(train_X, train_y)
        train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)
        train_losses = []
        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_loss = []
            for batch in train_loader:
                x, label = batch
                pred = model(x.to(device))
                loss = criterion(pred, label.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = sum(train_loss) / len(train_loss)
            train_losses.append(train_loss)
            if epoch % 5 == 0:
                print("epoch [%d], train MSE loss: [%f]" % (epoch, train_loss))
        end_time = time.time()
        avg_train_time += end_time - start_time

        valid_set = Data.TensorDataset(valid_X, valid_y)
        valid_loader = Data.DataLoader(valid_set, batch_size=1, shuffle=False)

        model.eval()
        valid_losses = []
        preds = []
        for batch in tqdm(valid_loader):
            x, true_y = batch
            with torch.no_grad():
                pred = model(x.to(device))
            loss = criterion(pred, true_y.to(device))
            valid_losses.append(loss)
            preds.append(pred)
        valid_loss = sum(valid_losses) / len(valid_losses)
        print("validation MSE loss: [%f]\n" % (valid_loss))
        avg_valid_loss += valid_loss

        if i + 1 == 4 and verbose:
            show_pred_func(valid_X, valid_y, preds)

    avg_train_time = avg_train_time / 4
    avg_valid_loss = avg_valid_loss / 4
    print("average training time: [%f], average validaiton loss: [%f]" % (avg_train_time, avg_valid_loss))
    return avg_train_time, avg_valid_loss, train_losses