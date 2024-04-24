from dqn import DQNAgent
from env import MazeEnv
import torch
import time

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    env = MazeEnv(render=True)
    input_dim = 7  # Example: 5 rays + 2 goal info (distance & angle)
    action_dim = 4  # forward, left, right, back
    agent = DQNAgent(input_dim=input_dim, action_dim=action_dim)
    agent.model.load_state_dict(torch.load('yitian/dqn_model_90.pth'))
    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)            
            next_state, reward, done = env.step(action)
            #agent.remember(state, action, reward, next_state, done)
            time.sleep(0.001)
            #agent.train_step()
            
            state = next_state
            total_reward += reward

            
            # print(reward)
            
            if env.steps > 2000: 
                break
                    
        print(f"Episode {episode+1}, Total reward: {total_reward}, Expl rate: {agent.exploration_rate} Final Dist {next_state[-2]} Steps {env.steps}")
        
        #if episode % 10 == 0:
            #agent.save(episode)
        
    env.close()

