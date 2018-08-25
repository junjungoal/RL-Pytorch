import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNet(nn.Module):
    def __init__(self, n_action):
        super(DuelingQNet, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1_adv = nn.Linear(7*7*64, 256)
        self.fc1_val = nn.Linear(7*7*64, 256)

        self.fc2_adv = nn.Linear(256, 1)
        self.fc2_val = nn.Linear(256, n_action)

    def forward(self, x):
        t = F.relu(self.conv1(x))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = t.view(t.size(0), -1)
        val = F.relu(self.fc1_val(t))
        adv = F.relu(self.fc1_adv(t))
        val = self.fc2_val(val).expand(t.size(0), self.n_action)
        adv = self.fc2_adv(adv)
        t = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_action)
        return t
