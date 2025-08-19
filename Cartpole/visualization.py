import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_cartpole_final.zip")
env = gym.make("CartPole-v1", render_mode="human")  # Display

for ep in range(5):
    obs, info = env.reset()
    done, truncated = False, False
    ep_r = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env.step(action)
        ep_r += r
    print(f"Episode {ep+1}: reward={ep_r}")

env.close()
