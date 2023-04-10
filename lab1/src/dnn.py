import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, layer=3, hidden_size=32, activation="relu"):
        super(DNN, self).__init__()
        input_size = 1
        output_size = 1
        self.dnn = nn.Sequential()

        self.activation = nn.ReLU()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()

        for i in range(1, layer + 1):
            if i == 1:
                self.dnn.add_module("hidden_layer1", nn.Linear(input_size, hidden_size))
                self.dnn.add_module(activation + str(i), self.activation)
                continue

            if i == layer:
                self.dnn.add_module("output_layer", nn.Linear(hidden_size, output_size))
                continue

            self.dnn.add_module("hidden_layer" + str(i), nn.Linear(hidden_size, hidden_size))
            self.dnn.add_module(activation + str(i), self.activation)

    def forward(self, x):
        x = self.dnn(x)
        return x
