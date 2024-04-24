from dqn import DQNAgent
from turtlebot_maze_env import Maze
import matplotlib.pyplot as plt

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    env = Maze(180, visuals=False, thresh=0.1)
    env_path = 'data/envs/env_2.txt'
    env.reset()
    env.loadENV(env_path)
    input_dim = 7  # Example: 5 rays + 2 goal info (distance & angle)
    action_dim = 4  # forward, left, right, back
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim)
    episodes = 500
    score = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if episode % 20 == 0 and episode > 0: # render every 10 episodes
                env.visuals=True
                env.reset()
                env.loadENV(env_path)
            if episode % 20 == 1 and episode > 0: # render every 10 episodes
                env.visuals=False
                env.reset()
                env.loadENV(env_path)
            action = agent.act(state)
                        
            next_state, reward, done = env.step(action, discr=True)
            agent.remember(state, action, reward, next_state, done)
            
            agent.train_step()
            
            state = next_state
            total_reward += reward
            score += env.goalAngle()[1]
            
            # print(reward)
            
            if env.steps > 500: 
                break
                    
        print(f"Episode {episode+1}, Total reward: {total_reward}, Expl rate: {agent.exploration_rate} Final Dist {next_state[-2]} Steps {env.steps}")
        
        if episode % 10 == 0:
            agent.save(episode)
            print("save model")
            plt.plot(range(1, len(score) + 1), score)
            plt.xlabel('Iter')
            plt.ylabel('Avg Score')
            plt.title(f'Scores')
            plt.grid(True)
            plt.savefig('scores.png')
        
    env.close()

