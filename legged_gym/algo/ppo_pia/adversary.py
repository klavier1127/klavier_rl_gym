import torch.nn as nn


class AdversaryNet(nn.Module):
    def __init__(self, env_features, hidden_size=32, num_layers=2, epsilon=0.1):
        super(AdversaryNet, self).__init__()
        self.epsilon = epsilon
        layers = []
        input_size = env_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = hidden_size
        layers.append(nn.Linear(hidden_size, env_features))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        delta = self.net(x) * self.epsilon
        return delta