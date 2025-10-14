import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, hidden_units=128):
        super().__init__()
        # one hidden layer have
        # First fully connected layer
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, x):
        # flatten the input image
        x = x.view(-1, 28 * 28)
        # apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        #  apply second fully connected layer
        x = self.fc2(x)
        return x
