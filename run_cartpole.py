# Python libraries
import sys
import gym
import numpy as np
import random as random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from a2c_simple import A2CAgent
import os
import pandas as pd

##-----------##
## Variables ##
##-----------##

## General parameters ##
USE_GPU = False # Whether or not to use the GPU for training
RENDER = True # Whether or not to save a render of every timestep into the images/ directory

SAVE_WEIGHTS = False
LOAD_WEIGHTS = False

## Neural Networks architecture Parameters ##
    # Dense layers
n_dense_layer_neurons = 24 # Number of neurons in each of the fully connected layers
dense_activation_function = "relu"
fc_args = [n_dense_layer_neurons, dense_activation_function] # Combine the convulational layer args

## Learning Parameters ##
gamma = 0.99 # The discount factor
actor_lr = 0.0001 # The learning rate for the Actor in A2C
critic_lr = 0.0005 # The learning rate for the Critic in A2C
learning_rates = [actor_lr, critic_lr]

N_EPISODES = 2000

##-----------##
## Utilities ##
##-----------##

# Disable GPU if needed
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##---------------##
## Training loop ##
##---------------##

## Environment setup ##
env = gym.make('CartPole-v0')

# Extract the size of the action and observation space
action_size = env.action_space.n
observation_size = env.observation_space.shape[0]
    
# Create the A2C agent
agent = A2CAgent(observation_size, action_size, learning_rates, fc_args, gamma)

# Keep track of the rewards and lengths of all episodes
episode_rewards = []
scores, episodes = [], []

for episode in range(N_EPISODES):
    done = False
    cur_episode_reward = 0
    current_observation = env.reset()
    current_observation = np.reshape(current_observation, [1, observation_size])

    # Run one episode
    while not done:
        if RENDER:
            env.render()

        # Retrieve the action of the agent based on the current observation
        action = agent.get_action(current_observation)
        # Take a step in the environment to obtain the reward and new observation
        next_observation, reward, done, info = env.step(action)
        # Reshape the next observation
        next_observation = np.reshape(next_observation, [1, observation_size])

        # Update the A2C agent based on the previous and next observations, the chosen action and the reward
        agent.train_model(current_observation, action, reward, next_observation, done)

        # Add this reward to the current reward of the episode
        cur_episode_reward += reward
        # Update the current observation
        current_observation = next_observation
        
        # If the episode has ended save the reward
        if done:
            # Save the reward of this episode
            episode_rewards.append(cur_episode_reward)
            # Report the reward of this episode
            print(f"[{episode}] Episode rewards: {cur_episode_reward}")

            # Save the score every 10 episodes
            if(episode % 10 == 0):
                d = {'Episodes': np.array(range(len(episode_rewards))), 'Rewards': episode_rewards}
                df = pd.DataFrame(d)
                df.to_csv(f'results/cartpole.csv', index=False)

print(f"Finished {N_EPISODES} episodes")