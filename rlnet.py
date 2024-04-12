import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.tan = nn.Tanh()
        self.device = device
    def forward(self, x):
        x = self.l1(x)
        x = self.tan(x)
        x = self.l2(x)
        x = self.tan(x)
        x = x/2
        mat = torch.tensor([[0.2, 0.1], [0.1, 0.2]])
        mat = mat.to(self.device)
        distr = torch.distributions.MultivariateNormal(x, mat)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs
    

class Policy2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(Policy2, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 128)
        self.l3 = nn.Linear(128, 32)
        self.l4 = nn.Linear(32, output_size)
        self.tan = nn.Tanh()
        self.device = device
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.tan(x)
        x = x/2
        mat = torch.tensor([[0.2, 0.1], [0.1, 0.2]])
        mat = mat.to(self.device)
        distr = torch.distributions.MultivariateNormal(x, mat)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs
class Policy3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(Policy3, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 128)
        self.l3 = nn.Linear(128, 32)
        self.l4 = nn.Linear(32, output_size)
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.device = device
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.sig(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        #set to 0.5 for default dropout rate
        self.p = 0.5
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(self.p),
        nn.Linear(1280, 1024),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(1024, 896),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(896, 768),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(768, 512),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(512, 384),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(384, 256),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(256, 256),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(256, 128),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(128, 64),nn.PReLU(), nn.Dropout(self.p),
        nn.Linear(64, 32),nn.PReLU(),
        nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out