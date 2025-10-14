import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)            # flatten
        x = F.relu(self.fc1(x))               # [batch_size, 128]
        x = self.fc2(x)                       # [batch_size, 10]
        return x
