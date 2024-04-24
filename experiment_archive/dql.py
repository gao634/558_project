import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import turtlebot_maze_env as maze
import os
import argparse

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        return self.l2(x)  # Removed softmax in output layer for Q-value estimation

    def predict(self, state):
        with torch.no_grad():
            return self(state)
        
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.uint8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, 1), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action  # Adjust based on discrete or not
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (torch.tensor(self.state_memory[batch]).float(),
                torch.tensor(self.action_memory[batch]),
                torch.tensor(self.reward_memory[batch]).float(),
                torch.tensor(self.new_state_memory[batch]).float(),
                torch.tensor(self.terminal_memory[batch]).float())

class DDQNAgent:
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.999995, epsilon_end=0.01, mem_size=25000, fname='ddqn_model.pth', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.brain_eval = DQN(input_dims, n_actions)
        self.brain_target = DQN(input_dims, n_actions)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.brain_eval.to(self.device)
        self.brain_target.to(self.device)
        self.optimizer = optim.Adam(self.brain_eval.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.replace_target_cnt = replace_target
        self.learn_step_counter = 0
        self.fname=fname

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            actions = self.brain_eval.predict(state)
            action = torch.argmax(actions).item()
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
            indices = np.arange(self.batch_size)
            model_output = self.brain_eval(states) # 64, 5
            q_pred = []
            for i, state in enumerate(states):
                q_pred.append(model_output[i][actions[i][0].item()])
            q_pred = torch.stack(q_pred)
            #q_pred = self.brain_eval(states)[indices, actions]
            model_output = self.brain_target(states_)
            q_next = []
            for i in range(len(states_)):
                q_next.append(model_output[i].max())
            #q_next = self.brain_target(states_).max(1)[0]
            q_next = torch.stack(q_next)
            q_target = rewards + self.gamma * q_next
            loss = self.criterion(q_pred, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.learn_step_counter % self.replace_target_cnt == 0:
                self.brain_target.load_state_dict(self.brain_eval.state_dict())

            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
            self.learn_step_counter += 1

    def save_model(self):
        torch.save(self.brain_eval.state_dict(), self.fname)

    def load_model(self):
        self.brain_eval.load_state_dict(torch.load(self.fname))
        self.brain_target.load_state_dict(torch.load(self.fname))

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


TOTAL_GAMETIME = 500
N_EPISODES = 1000
REPLACE_TARGET = 10

game = maze.Maze(180, visuals=False, thresh=0.1)
env_path = 'data/envs/env_2.txt'
game.reset()
game.loadENV(env_path)

GameTime = 0 
GameHistory = []
renderFlag = False

ddqn_agent = DDQNAgent(alpha=0.0001, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=2,fname='test_models/model.h5')

#ddqn_agent.load_model()
#ddqn_agent.update_network_parameters()

ddqn_scores = []
eps_history = []
start = (1.5, 1.5)
goal = (2.5, 1.5)
angles = []
def run():

    for e in range(N_EPISODES):
        

        score = 0
        counter = 0
        avg_angle = 0
        
        observation_, reward, done = game.step()
        observation = observation_

        gtime = 0 # set game time back to 0
        # if you want to render every episode set to true
        
        if e % 20 == 0 and e > 0: # render every 10 episodes
            game.visuals=True
            game.reset()
            game.loadENV(env_path)
        if e % 20 == 1 and e > 0: # render every 10 episodes
            game.visuals=False
            game.reset()
            game.loadENV(env_path)
        game.setPos(start[0], start[1], -np.pi, False)
        game.setGoal(goal)
        while not done:

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action, discr=True)

            score += reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            avg_angle += abs(game.goalAngle()[0])
            
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])
        avg_angle /= gtime
        angles.append(avg_angle)
        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            plt.plot(range(1, len(angles) + 1), angles)
            plt.xlabel('Iter')
            plt.ylabel('Avg Score')
            plt.title(f'Scores')
            plt.grid(True)
            plt.savefig('scores.png')
            
        print('episode: ', e,'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' average angle %.2f' % avg_angle,
              ' epsilon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

run()        