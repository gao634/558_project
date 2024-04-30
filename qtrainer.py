from dqn import DQNAgent
from turtlebot_maze_env import Maze
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    env = Maze(180, visuals=True, thresh=0.2)
    env_path = 'data/envs/env_4.txt'
    env.reset()
    env.loadENV(env_path)
    input_dim = 9  # Example: 5 rays + 2 goal info (distance & angle) + 2 vel
    action_dim = 3  # forward, left, right, back
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim, gamma = 0.2, eps=.7, decay=0.99999)
    #agent.model.load_state_dict(torch.load('./models/archive/m6.pth')) 
    #agent.model.load_state_dict(torch.load('./test_models/dqn_model_800.pth'))
    agent.target_model.load_state_dict(agent.model.state_dict())
    episodes = 1500
    max_steps = 1000
    scores = []
    angles = []
    start = (1, 1)
    goal = (2, 1)
    for episode in range(episodes):
        state, reward, done = env.step()
        total_reward = 0
        done = False
        score = 0
        # if episode % 20 == 0 and episode > 0: # render every 10 episodes
        #     env.visuals=True
        #     env.reset()
        #     env.loadENV(env_path)
        # if episode % 20 == 1 and episode > 0: # render every 10 episodes
        #     env.visuals=False
        #     env.reset()
        #     env.loadENV(env_path)
        #env.setPos(start[0], start[1], 0, True)
        env.setPos(start[0], start[1])
        r = np.random.random() * 3 + 1.5
        goal = (r, 1)
        env.setGoal(goal)
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action, discr=True)
            agent.remember(state, action, reward, next_state, done)
            #print(state)
            
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            score += env.goalAngle()[1]
            
            # print(reward)
            #print(env.steps)
            if env.steps > max_steps: 
                break
                    
        print(f"Episode {episode+1}, Total reward: {total_reward}, Expl rate: {agent.exploration_rate} Final Dist {next_state[-3]} Steps {env.steps}")
        scores.append(score/env.steps)
        e = episode + 1
        agent.update_target_model()
        if e % 10 == 0:
            plt.plot(range(1, len(scores) + 1), scores)
            plt.xlabel('Iter')
            plt.ylabel('Avg Score')
            plt.title(f'Scores')
            plt.grid(True)
            plt.savefig('scores.png')
        if e % 50 == 0:
            agent.save(e)
            env.reset()
            env.loadENV(env_path)
        

