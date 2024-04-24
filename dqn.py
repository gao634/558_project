import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, batch_size=128, eps = 1, decay = 0.9999, buffer_size=10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(input_dim, action_dim).float()
        self.target_model = DQN(input_dim, action_dim).float()
        self.target_model.load_state_dict(self.model.state_dict())
        self.brain_eval.to(self.device)
        self.brain_target.to(self.device)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.exploration_rate = eps
        self.exploration_decay = decay
        self.exploration_min = 0.01

    def act(self, state):
        """ Choose an action based on the current state """
        if random.random() < self.exploration_rate:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = state.to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        #states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Get current Q value
        current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Get next Q value from target model
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Compute loss
        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        
        # print(f'Loss {loss.item()}, Exploration {self.exploration_rate}')
        
        # Periodically update target model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, steps):
        path = f"./test_models/dqn_model_{steps}.pth"
        print('Save model to ', path)
        torch.save(self.model.state_dict(), path)
