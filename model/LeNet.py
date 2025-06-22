import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, feature=False):
        x = F.relu(self.conv1(x))    # input(1, 28, 28) output(16, 24, 24)
        x = self.pool1(x)            # output(16, 12, 12)
        x = F.relu(self.conv2(x))    # output(32, 8, 8)
        x = self.pool2(x)            # output(32, 4, 4)

        feat = x if feature else None
        x = x.view(x.size(0), -1)    # output(512)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x, feat

class Simple_LeNet(nn.Module):
    def __init__(self):
        super(Simple_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)      # (1,28,28) -> (16,24,24)
        self.pool1 = nn.MaxPool2d(4, 4)       # (16,24,24) -> (16,12,12)
        self.fc1 = nn.Linear(16*6*6, 10)    # 只一个全连接，输出10类

    def forward(self, x, feature=False):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        feat = x if feature else None
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x, feat
