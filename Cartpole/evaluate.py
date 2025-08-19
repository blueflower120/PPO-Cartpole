import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("ppo_cartpole_final.zip")
env = gym.make("CartPole-v1")

rewards = []
for ep in range(100):
    obs, info = env.reset()
    done, truncated = False, False
    ep_r = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)  # Give an action based on the observation
        # deterministic=True means choosing the best action
        obs, r, done, truncated, info = env.step(action)  # State in the next step
        ep_r += r
    rewards.append(ep_r)
    print(f"Episode {ep+1}: reward={ep_r}")

env.close()

rewards = np.array(rewards)
print("="*50)
print(f"Mean: {rewards.mean():.2f}, Min: {rewards.min()}, Max: {rewards.max()}, "
      f"500-rate: {(rewards==500).mean():.2%}")
