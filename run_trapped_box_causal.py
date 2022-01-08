# Python libraries
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
import os, glob
# SSD library
from social_dilemmas.envs.agent import BASE_ACTIONS, APPLE_ACTIONS
from social_dilemmas.envs.apple_learning import AppleLearningEnv
from social_dilemmas.envs.trapped_box import TrappedBoxEnv
# Own imports
from a2c_conv import *

##-----------##
## Variables ##
##-----------##

## General parameters ##
USE_GPU = False # Whether or not to use the GPU for training
    # Rendering images
RENDER = True # Whether or not to save a render of every timestep into the images/ directory
RENDER_DIR = "images/trapped_box/"
    # Weights
SAVE_WEIGHTS_DIR = "model/trapped_box/"
SAVE_WEIGHTS = False
LOAD_WEIGHTS = False

# Parameters from social influences paper
n_agents = 2
state_size = (1, 15, 15, 3)

## Experiment related parameters ##
N_EPISODES = 100000
N_STEPS = 100

## Neural Networks architecture Parameters ##
    # Convolutional layer
n_output_conv = 6 # Number of output channels of the convolutional layer
kernel_size = 3 # Kernel size of the convolutional layer
strides=(1, 1) # The stride of the convolutional layer
conv_activation_function = "linear"
conv_args = [n_output_conv, kernel_size, strides, conv_activation_function] # Combine the convulational layer args
    # Dense layers
n_dense_layer_neurons = 32 # Number of neurons in each of the fully connected layers
dense_activation_function = "linear"
fc_args = [n_dense_layer_neurons, dense_activation_function] # Combine the convulational layer args

## Learning Parameters ##
gamma = 0.99 # The discount factor
actor_lr = 0.001 # The learning rate for the Actor in A2C
critic_lr = 0.005 # The learning rate for the Critic in A2C
learning_rates = [actor_lr, critic_lr]

##-----------##
## Utilities ##
##-----------##

# Disable GPU if needed
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if RENDER:
    # Delete all previously generated images (save them before re-running the file)
    files = glob.glob(f"/home/liguedino/Documents/github/project_comp_game_theory/{RENDER_DIR}/*")
    for f in files:
        print(f"Image: {f}")
        os.remove(f)

##---------------##
## Training loop ##
##---------------##

episode_rewards = np.zeros(N_EPISODES)
episode_length = np.zeros(N_EPISODES)

## Environment setup ##
env = TrappedBoxEnv(num_agents=n_agents) # Create the environment
env.setup_agents() # Setup the agents
agents = env.agents # Retrieve the newly created agents

## A2C setup ##
A2C_agents = {} # Keep track of the newly created A2C agents
# Create an A2C agent for each agent in the environment
for agentKey in agents.keys():
    A2C_agents[agentKey] = A2CAgent(state_size, env.n_actions, learning_rates, conv_args, fc_args, gamma)

# Training loop
for episode in range(N_EPISODES):
    # Retrieve the initial state of the environment
    states = env.reset()
    # Keep a queue of observations for every agent (to be able to provide a history of observations as an observation to the agent)
    agentQueus = {}

    for agentKey in A2C_agents.keys():
        state = states[agentKey]["curr_obs"]
        # state = np.reshape(state, [1, 15, 15, 3])
        # Create a queue for each agent
        queue = deque()
        # Add the initial observation to the queue
        queue.extendleft([state])
        # Save the queue
        agentQueus[agentKey] = queue

    cumulative_reward = 0.
    episode_done = False

    for step in range(N_STEPS):
        # If RENDER is true save an image of every timestep to the images/ directory
        if RENDER:
            env.render(f"{RENDER_DIR}/{str(episode).zfill(10)}_{str(step).zfill(10)}.png")
        # Keep track of the actions of each agent
        curr_actions = {}
        # For every agent: choose an action based on the current observation
        for agentKey, agentObject in A2C_agents.items():
            queue = agentQueus[agentKey]
            current_observations = tf.concat([x for x in queue], 0)
            current_observations = np.reshape(current_observations, [1, 1, 15, 15, 3])
            action = agentObject.get_action(current_observations)
            curr_actions[agentKey] = action

        # Run one step in the environment 
        next_states, rewards, dones, _info = env.step(curr_actions) # agent id
        
        ##-------------------##
        ## Update the agents ##
        ##-------------------##
        
        for agentKey, agentObject in A2C_agents.items():
            next_state = next_states[agentKey]["curr_obs"]

            # Extract this agent's reward
            reward = rewards[agentKey]
            # Update the cumulative reward
            cumulative_reward += reward
            # Extract whether or not the environment indicates that this episode is done
            done = dones[agentKey]
            # Extract the chosen action of the agent
            action = curr_actions[agentKey]

            # Extract the previous observations of this agent
            queue = agentQueus[agentKey]
            # Reshape the previous observations
            prev_observations = tf.concat([x for x in queue], 0)
            prev_observations = np.reshape(prev_observations, [1, 1, 15, 15, 3])
            # Remove the oldest observation from the queue
            queue.pop()
            # Add the new observations to the queue
            queue.extendleft([next_state])
            # Reshape the new observations
            allstates = tf.concat([x for x in queue], 0)
            allstates = np.reshape(allstates, [1, 1, 15, 15, 3])

            # Update the A2C agent based on the previous and next observations, the chosen action and the reward
            agentObject.train_model(prev_observations, action, reward, allstates, done)

    # Save the length of this episode
    episode_length[episode] = step
    # Save the cumulative reward of this episode
    episode_rewards[episode] = cumulative_reward

    print(f"[{episode}] Episode rewards: {cumulative_reward} after {step+1} steps")

    if (episode % 10) == 0:
        d = {'Episodes': np.array(range(len(episode_rewards))), 'Rewards': episode_rewards}
        df = pd.DataFrame(d)
        df.to_csv('results/trapped_box_causal.csv', index=False)

print(f"Finished {N_EPISODES} episodes")


