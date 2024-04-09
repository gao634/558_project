import visualizer3d as vis
import lidar
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.tan = nn.Tanh()
        self.opt = torch.optim.Adam()
    def forward(self, x):
        x = self.l1(x)
        x = self.tan(x)
        x = self.l2(x)
        x = self.tan(x)
        mat = torch.tensor([[0.2, 0.1], [0.1, 0.1]])
        distr = torch.distributions.MultivariateNormal(x, mat)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs
    def train(env, start, end):
        max_iters = 100
        for iter in range(max_iters):
            p.stepSimulation()
    def reward():
        pass
    def loss():
        pass

def main(args):
    input_size = 180 + 4 + 100


    p.connect(p.GUI)
    id_list = vis.loadENV('data/envs/env_0.txt')
    id = vis.loadAgent(0.5, 0.5)
    #p.loadURDF('assets/cylinder.urdf', [1.5, 0.5, 3])
    p.loadURDF('assets/ground.urdf', [0, 0, -0.1])
    p.setGravity(0, 0, -9.81) 
    vel = 0
    for epoch in args.num_epochs:
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--num-episodes', type=int, default=100)
parser.add_argument('--num-episodes', type=int, default=100)
parser.add_argument('--load-path', type=str, default='100ep.dat')
parser.add_argument('--save-path', type=str, default='1001ep.dat')
parser.add_argument('--plot', default=True, action='store_true')
parser.add_argument('--save', default=True, action='store_true')
parser.add_argument('--visuals', default=True, action='store_true')

args = parser.parse_args()

print(args)
main(args=None)