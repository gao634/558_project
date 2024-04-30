from dqn import DQNAgent
from turtlebot_maze_env import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader import dataloader


if __name__ == "__main__":
    env = Maze(180, visuals=True, thresh=0.2)
    env_path = 'data/envs/env_6.txt'
    input_dim = 9  # Example: 5 rays + 2 goal info (distance & angle) + 2 vel
    action_dim = 3  # forward, left, right, back
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim, eps=0, min_eps = 0, decay=0.999995)
    #agent.model.load_state_dict(torch.load('./test_models/dqn_model_3000.pth'))
    agent.model.load_state_dict(torch.load('./models/archive/m6.pth'))
    start_path = 0
    num_paths = 10
    successes = []
    for i in range(start_path, start_path + num_paths):
        env.reset()
        env.loadENV(env_path)
        file = 'data/env6/path' + str(i) + '.txt'
        path = np.loadtxt(file, np.float32)
        start = path[0]
        goal = path[1]
        env.setGoal(goal)
        env.setPos(start[0], start[1], env.goalWorldAngle(start[0], start[1]), False)
        state, reward, done = env.step()
        save_state = state
        for j in range(1, len(path)):
            goal = path[j]
            env.setGoal(goal)
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action, discr=True)
                
                state = next_state
                if env.steps > 1000: 
                    break
            if next_state[-3] < env.thresh:
                env.steps = 0
                if j == len(path) - 1:
                    successes.append(True)
            else:
                successes.append(False)
                break

        

