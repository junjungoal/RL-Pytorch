import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, n_action):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(3136, 256)
        self.linear2 = nn.Linear(256, n_action)

    def forward(self, x):
        t = F.relu(self.conv1(x))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = t.view(t.size(0), -1)
        t = F.relu(self.linear1(t))
        t = self.linear2(t)
        return t
