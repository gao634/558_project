import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse
import rlnet
import turtlebot_maze_env as maze
import os
import prm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 180 + 100
    hidden_size = 256
    output_size = 3
    gamma = 0.9
    learning_rate = 0.001

    env = maze.Maze(180, visuals=False, thresh=0.1)
    model = rlnet.Policy3(input_size, hidden_size, output_size, device)
    if args.start_epoch > 0:
        mp = args.model_path + 'epoch' + str(args.start_epoch) + '.dat'
        model.load_state_dict(torch.load(mp))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossF = torch.nn.MSELoss()
    scores = []
    env_path = 'data/envs/env_0.txt'
    env.reset()
    env.loadENV(env_path)
    env_tensor = env.getEnvTensor()
    num_episodes = 20
    map = prm.PRM(tree=False, geom='point')
    map.env.load('data/envs/env_0.txt')
    for epoch in range(args.epochs):
        avg_score = 0
        i = 0
        iters = 1000
        ax = []
        ay = []
        ah = []
        pay = []
        pax = []
        pah = []
        while i < iters:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 10)
            h = np.random.uniform(np.pi * 2)
            if map.collision((x, y)):
                continue
            i += 1
            env.setPos(x, y, h)
            input = torch.tensor(env.getInput())
            input = torch.cat((input, env_tensor))
            input = input.to(device)
            px, py, ph = model(input)
            px = px * 10
            py = py * 10
            ph = ph * np.pi * 2
            ax.append(x)
            ay.append(y)
            ah.append(h)
            pax.append(px)
            pay.append(py)
            pah.append(ph)
        ax = torch.tensor(ax, requires_grad=True)
        pax = torch.tensor(pax, requires_grad=True)
        ay = torch.tensor(ay, requires_grad=True)
        pay = torch.tensor(pay, requires_grad=True)
        ah = torch.tensor(ah, requires_grad=True)
        pah = torch.tensor(pah, requires_grad=True)
        lossx = lossF(ax, pax)
        lossy = lossF(ay, pay)
        lossh = lossF(ah, pah)
        loss = lossx + lossy + lossh
        #gradient calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scores.append(loss.item())
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}')           



        curr_epoch = epoch + 1 + args.start_epoch
        if (curr_epoch) % 50 == 0:
            # save model
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            save_path = args.model_path + 'epoch' + str(curr_epoch) + '.dat'
            torch.save(model.state_dict(), save_path)
            if args.plot:
                plt.plot(range(1, len(scores) + 1), scores)
                plt.xlabel('Iter')
                plt.ylabel('Avg Score')
                plt.title(f'Scores')
                plt.grid(True)
                plt.savefig('scores.png')
        #plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--data-path', type=str, default='./data/')

parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--plot', default=True, action='store_true')

args = parser.parse_args()

print(args)
main(args)