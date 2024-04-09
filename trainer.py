import visualizer2d as vis2
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse
from rlnet import Policy
import turtlebot_maze_env as maze
import os

def main(args):
    input_size = 180 + 4 + 100
    hidden_size = 256
    output_size = 2

    env = maze.Maze(180)
    model = Policy(input_size, hidden_size, output_size)
    if args.start_epoch > 0:
        mp = args.model_path + 'epoch' + str(args.start_epoch) + '.dat'
        model = model.load_state_dict(torch.load(mp))

    for epoch in args.epochs:
        for m in range(args.N):
            env_path = args.data_path + 'envs/env_' + str(m) + '.txt'
            env.reset()
            env.loadENV(env_path)
            file_path = args.data_path + 'env' + str(m) + '/path'
            for n in range(args.NP):
                file_path += str(n) + '.txt'
                data = vis2.getPathData(file_path)
                for i in range(1, len(data)):
                    start = data[i-1]
                    goal = data[i]
                    env.setPos(start)
                    env.setGoal(goal)
                    lidar, reward, terminated, collision = env.step()
                    steps = 0
                    while not collision and not terminated and steps < 100:
                        # training loop
                        env_tensor = env.getEnvTensor()
                        input = torch.stack(torch.tensor(lidar), env_tensor, torch.tensor(start))
                        actions, logprobs = model(input)
                        lidar, reward, terminated, collision = env.step(actions)
                        steps += 1



        curr_epoch = epoch + 1 + args.start_epoch
        if (curr_epoch) % 100 == 0:
            # save model
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            save_path = args.model_path + 'epoch' + str(curr_epoch) + '.dat'
            torch.save(model.state_dict(), save_path)
    scores = []
    if args.plot:
        plt.plot(range(1, len(scores) + 1), scores)
        plt.xlabel('Iter')
        plt.ylabel('Avg Score')
        plt.title(f'Number of episodes = {args.num_episodes}')
        plt.grid(True)
        plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--data-path', type=str, default='./data/')
parser.add_argument('--N', type=int, default=1, help='number of environments')
parser.add_argument('--NP', type=int, default=20, help='number of paths per environment')

parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--plot', default=True, action='store_true')

args = parser.parse_args()

print(args)
main(args)