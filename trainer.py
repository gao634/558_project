import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse
from rlnet import Policy
import turtlebot_maze_env as maze
import os

def getPathData(file):
    data = np.loadtxt(file, np.float32)
    return data

def lossF(data, gamma, num_episodes, device):
    loss = 0
    for rewards, log_probs in data:
        # Compute discounted returns using vectorized operations
        rewards = torch.stack(rewards)
        log_probs = torch.stack(log_probs)
        T = len(rewards)
        discounts = torch.pow(gamma, torch.arange(T))
        discounted_returns = []
        for i in range(T):
            discounted_returns.append(torch.dot(discounts, rewards))
            discounts = discounts[0:-1]
            rewards = rewards[1:]
        discounted_returns = torch.stack(discounted_returns)

        # Compute baseline and standard deviation
        baseline = torch.mean(discounted_returns)
        stdev = torch.std(discounted_returns)
        if len(discounted_returns) == 1:
            stdev = 1
        # Compute advantage and accumulate weighted log probabilities
        advantages = (discounted_returns - baseline) / (stdev + 0.00001)
        advantages = advantages.to(device)
        if advantages.dim() == 0:
            advantages = advantages.unsqueeze(0)
        loss -= torch.dot(log_probs, advantages)

    return loss / num_episodes
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 180 + 2 + 100
    hidden_size = 256
    output_size = 2
    gamma = 0.9
    learning_rate = 0.01

    env = maze.Maze(180, visuals=False)
    model = Policy(input_size, hidden_size, output_size, device)
    if args.start_epoch > 0:
        mp = args.model_path + 'epoch' + str(args.start_epoch) + '.dat'
        model = model.load_state_dict(torch.load(mp))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scores = []
    for epoch in range(args.epochs):
        loss_data = []
        avg_score = 0
        avg_steps = 0
        num_episodes = 0
        for m in range(args.N):
            env_path = args.data_path + 'envs/env_' + str(m) + '.txt'
            env.reset()
            env.loadENV(env_path)
            env_tensor = env.getEnvTensor()
            for n in range(args.NP):
                print('path ', n)
                file_path = args.data_path + 'env' + str(m) + '/path'
                file_path += str(n) + '.txt'
                data = getPathData(file_path)
                score = 0
                steps = 0
                num_episodes += len(data) - 1
                for i in range(1, len(data)):
                    print('segment ', i)
                    start = data[i-1]
                    goal = data[i]
                    env.setPos(start[0], start[1])
                    env.setGoal(goal)
                    lidar, reward, terminated, collision = env.step()
                    steps = 0
                    rewards = []
                    probs = []
                    while not collision and not terminated and steps < 100:
                        # training loop
                        input = torch.cat([torch.tensor(lidar), env_tensor])
                        input = torch.cat([input, torch.tensor(start)])
                        input = input.to(device)
                        actions, log_prob = model(input)
                        lidar, reward, terminated, collision = env.step((actions[0].item(), actions[1].item()))
                        rewards.append(torch.tensor(reward, dtype=torch.float32))
                        probs.append(log_prob)
                        score += reward
                        steps += 1
                        if terminated:
                            print('success')
                    avg_score += score
                    avg_steps += steps
                    loss_data.append((rewards, probs))
                    #print(len(reward_probs))
        loss = lossF(loss_data, gamma, num_episodes, device)
        #gradient calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_score /= num_episodes
        avg_steps /= num_episodes
        print(avg_score, avg_steps)
        scores.append(avg_score)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}')           



        curr_epoch = epoch + 1 + args.start_epoch
        if (curr_epoch) % 100 == 0:
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
        plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--data-path', type=str, default='./data/')
parser.add_argument('--N', type=int, default=1, help='number of environments')
parser.add_argument('--NP', type=int, default=1, help='number of paths per environment')

parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--plot', default=True, action='store_true')

args = parser.parse_args()

print(args)
main(args)