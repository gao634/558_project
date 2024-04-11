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
        mat = torch.tensor([[0.2, 0.1], [0.1, 0.2]])
        mat = mat.to(self.device)
        distr = torch.distributions.MultivariateNormal(x, mat)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs