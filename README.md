# PPO-Cartpole
My reinforcement learning practice project: PPO for Cartpole-v1 environment. This project utilizes the proximal policy optimization (PPO) algorithm for training and testing in the CartPole-v1 environment, one of the classic introductory experiments in reinforcement learning. The goal is to make a cart keep a vertical pole upright by moving left and right. 

# File Description
  # train.py
  Train the model using PPO in the CartPole-v1 environment and save the final model as ppo_cartpole_final.zip
  # evaluate.py
  Load the trained model, run 100 test rounds, and output the score for each round and the average score.
  # visualization.py
  Load the model and run 5 games of visualization to show the performance of the agent.

# Experimental Results
After about 400,000 steps of training, the agent can consistently achieve full scores.
