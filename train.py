import gymnasium as gym  # Environment creation and interaction
from stable_baselines3 import PPO  # Standard algorithm library for RL
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, seed: int = 0, render: bool = False):
    def _thunk():
        env = gym.make(env_id, render_mode="human" if render else None)  # Display if render_mode="human"
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk  # Return a function

def main():
    env_id = "CartPole-v1"
    timesteps = 400_000
    seed = 42
    save_path = "ppo_cartpole_final.zip"
    tb_logdir = None

    # Reproducibility
    set_random_seed(seed)

    # Training environment
    train_env = DummyVecEnv([make_env(env_id, seed=seed, render=False)])
    train_env = VecMonitor(train_env)

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_logdir,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.98,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # Train to full timesteps
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save the final model
    model.save(save_path)
    print(f"[INFO] Saved final model to {save_path}")

if __name__ == "__main__":
    main()