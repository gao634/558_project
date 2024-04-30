from dqn import DQNAgent
from turtlebot_maze_env import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_loader import dataloader


if __name__ == "__main__":
    env = Maze(180, visuals=True, thresh=0.2)
    env_path = 'data/envs/env_6.txt'
    env.reset()
    env.loadENV(env_path)
    input_dim = 9  # Example: 5 rays + 2 goal info (distance & angle) + 2 vel
    action_dim = 3  # forward, left, right, back
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim, gamma = .2, eps=1, decay=0.999995)
    #agent.model.load_state_dict(torch.load('./test_models/dqn_model_2200.pth'))
    agent.model.load_state_dict(torch.load('./models/archive/m6.pth'))
    episodes = 5000
    scores = []
    data = dataloader(6, 1, 0, 190)
    score = 0
    for episode in range(episodes):
        state, reward, done = env.step()
        total_reward = 0
        done = False
        ind = np.random.randint(len(data))
        start, goal = data[ind]
        env.setGoal(goal)
        env.setPos(start[0], start[1], env.goalWorldAngle(start[0], start[1]), True)
        #print(env.goalWorldAngle(start[0], start[1]), env.getPos()[5])
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action, discr=True)
            agent.remember(state, action, reward, next_state, done)
            #print(state)
            
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            # print(reward)
            #print(env.steps)
            if env.steps > 500: 
                break
                    
        print(f"Episode {episode+1}, Total reward: {total_reward}, Segment {ind}, Expl rate: {agent.exploration_rate} Final Dist {next_state[-3]} Steps {env.steps}")
        e = episode + 1
        if total_reward > 100:
            score += 1
        agent.update_target_model()
        if e % 50 == 0:
            # score = 0
            # agent.exploration_rate = 0
            # for i in range(100):
            #     state, reward, done = env.step()
            #     total_reward = 0
            #     done = False
            #     ind = np.random.randint(len(data))
            #     start, goal = data[ind]
            #     env.setGoal(goal)
            #     env.setPos(start[0], start[1], env.goalWorldAngle(start[0], start[1]), True)
            #     #print(env.goalWorldAngle(start[0], start[1]), env.getPos()[5])
            #     while not done:
            #         action = agent.act(state)
            #         next_state, reward, done = env.step(action, discr=True)
            #         state = next_state
            #         total_reward += reward
                    
            #         # print(reward)
            #         #print(env.steps)
            #         if env.steps > 500: 
            #             break
            #     if total_reward > 600:
            #         score += 1
            scores.append(score)
            score = 0
            plt.plot(range(1, len(scores) + 1), scores)
            plt.xlabel('Iter')
            plt.ylabel('Avg Score')
            plt.title(f'Scores')
            plt.grid(True)
            plt.savefig('scores.png')
            #agent.exploration_rate = 0.05
        if e % 100 == 0:
            agent.save(e)
            env.reset()
            env.loadENV(env_path)
        

