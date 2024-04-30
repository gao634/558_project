from dqn import DQNAgent
from turtlebot_maze_env import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
import time


if __name__ == "__main__":
    env = Maze(180, visuals=True, thresh=0.1)
    env_path = 'data/envs/env_2.txt'
    env.reset()
    env.loadENV(env_path)
    input_dim = 9  # Example: 5 rays + 2 goal info (distance & angle) + 2 wheel vel
    action_dim = 3  # forward, left, right
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim, eps=0, min_eps = 0)
    agent.model.load_state_dict(torch.load('./models/archive/m5.pth'))
    #agent.model.load_state_dict(torch.load('./test_models/dqn_model_500.pth'))
    episodes = 20
    scores = []
    start = (1.5, 1.5)
    goal = (2.5, 1.5)
    for episode in range(episodes):
        state, reward, done = env.step()
        total_reward = 0
        done = False
        score = 0
        env.setPos(start[0], start[1])
        env.setGoal(goal)
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action, discr=True)
            
            state = next_state
            total_reward += reward
            
            score += env.goalAngle()[1]
            #time.sleep(0.001)
            # print(reward)
            #print(env.steps)
            if env.steps > 500: 
                break
                    
        print(f"Episode {episode+1}, Total reward: {total_reward}, Expl rate: {agent.exploration_rate} Final Dist {next_state[-1]} Steps {env.steps}")
        scores.append(score/env.steps)
        if episode % 10 == 0:
            agent.save(episode)
            plt.plot(range(1, len(scores) + 1), scores)
            plt.xlabel('Iter')
            plt.ylabel('Avg Score')
            plt.title(f'Scores')
            plt.grid(True)
            plt.savefig('scores.png')
        
    env.close()

