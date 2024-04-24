import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import torch
import torch.nn as nn
import argparse
import rlnet
import turtlebot_maze_env as maze
import os
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
def lossi(data, gamma, num_episodes, device):
    loss = 0
    for rewards, log_probs in data:
        probs = 0.0
        for j, reward in enumerate(rewards):
            returns = 0.0
            for k in range(len(rewards) - j):
                returns += gamma ** k * rewards[j+k]
            rewards[j] = returns
        rewards = torch.tensor(rewards)
        baseline = torch.mean(rewards)
        stdev = torch.std(rewards)
        for j, reward in enumerate(rewards):
            probs += log_probs[j] * (reward - baseline) / (stdev + 0.00001)
        loss -= probs
    return loss / num_episodes
def credit_loss(data, gamma, num_episodes):
    loss = 0
    for rewards, log_probs in data:
        probs = 0.0
        p = 0
        for j, reward in enumerate(rewards):
            returns = 0.0
            for k in range(len(rewards) - j):
                returns += gamma ** k * reward
            probs += log_probs[j] * returns
        loss -= probs
    return loss / num_episodes
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 4
    gamma = 0.99
    learning_rate = 0.005

    env = maze.Maze(180, visuals=True, thresh=0.1)
    model = rlnet.PolDiscrete(input_size)
    if args.start_epoch > 0:
        mp = args.model_path + 'epoch' + str(args.start_epoch) + '.dat'
        model.load_state_dict(torch.load(mp))
    model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    scores = []
    env_path = 'data/envs/env_2.txt'
    env.reset()
    env.loadENV(env_path)
    env_tensor = env.getEnvTensor()
    #file_path = 'data/env1/path0.txt'
    #data = getPathData(file_path)
    num_episodes = 50
    for epoch in range(args.epochs):
        avg_score = 0
        avg_angle = 0
        collision_rate = 0
        success_rate = 0
        truncate_rate = 0
        avg_steps = 0
        start = (1.5, 1.5)
        goal = (2.5, 1.5)
        loss_data = []
        for iter in range(num_episodes):
            env.setPos(start[0], start[1])
            env.setGoal(goal)
            #print(env.goalAngle())
            terminated, collision = env.step()
            steps = 0
            score = 0
            rewards = []
            probs = []
            iters = 300
            while not terminated and steps < iters:
                # training loop
                steps += 1
                x, y, z, rr, rp, ry = env.getPos()
                angle, distance = env.goalAngle()
                input = torch.tensor((angle, distance, env.getVel()[0], env.getVel()[1]))
                input = input.to(device)
                action, log_prob = model(input)
                terminated, collision = env.step(action, discr=True)
                reward = env.rewardF()
                #if steps == iters:
                    #reward = -10
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                probs.append(log_prob)
                score += reward
                if collision:
                    collision_rate += 1
                if terminated:
                    success_rate += 1
                #if terminated:
                #    print('success')
            if steps == iters:
                truncate_rate += 1
            avg_score += score
            avg_steps += steps
            avg_angle += 2 ** -(env.goalAngle()[0] ** 2)
            loss_data.append((rewards, probs))
            #print(len(reward_probs))
        loss = lossF(loss_data, gamma, num_episodes, device)
        #gradient calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_score /= num_episodes
        avg_steps /= num_episodes
        avg_angle /= num_episodes
        success_rate /= num_episodes
        truncate_rate /= num_episodes
        collision_rate /= num_episodes
        #print(success_rate, collision_rate, truncate_rate)
        print(avg_score, avg_steps, avg_angle)
        scores.append(avg_score)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}')           



        curr_epoch = epoch + 1 + args.start_epoch
        if epoch == 20:
            env.visuals = False
            env.reset()
            env.loadENV(env_path)
        if (curr_epoch) % 30 == 0:
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