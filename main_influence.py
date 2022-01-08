from social_dilemmas.envs import env_creator
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv
from social_dilemmas.envs.agent import BASE_ACTIONS, CLEANUP_ACTIONS, HARVEST_ACTIONS, TRAPPED_ACTIONS, APPLE_ACTIONS
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from a2c_conv import *
import pandas as pd
import tensorflow_probability as tfp
from random import randrange
from collections import deque
from social_dilemmas.envs.trapped_box import TrappedBoxEnv
from social_dilemmas.envs.apple_learning import AppleLearningEnv
from social_dilemmas.envs.agent import TRAPPED_ACTIONS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##------------------------------------------------##
## Unfinished code due to performance/time issues ##
##------------------------------------------------##

env = TrappedBoxEnv(num_agents=2)
env.setup_agents()

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

agents = env.agents

state_size = (1, 15, 15, 3)


A2C_agents = {}

for agentKey in agents.keys():
    A2C_agents[agentKey] = A2CAgent(state_size, env.n_actions, learning_rates, conv_args, fc_args, gamma) 
print("All agents(A2C)", agents)

##----------------------------##
## Delete all previous images ##
##----------------------------##

import os
import glob

# files = glob.glob('/home/liguedino/Documents/github/project_comp_game_theory/images/*')
# for f in files:
#     os.remove(f)

@staticmethod
def kl_div(x, y):
    """
    Calculate KL divergence between two distributions.
    :param x: A distribution
    :param y: A distribution
    :return: The KL-divergence between x and y. Returns zeros if the KL-divergence contains NaN
    or Infinity.
    """
    dist_x = tfp.distributions.Categorical(probs=x)
    dist_y = tfp.distributions.Categorical(probs=y)
    result = tfp.distributions.kl_divergence(dist_x, dist_y)

    # Don't return nans or infs
    is_finite = tf.reduce_all(tf.math.is_finite(result))

    def true_fn():
        return result
    def false_fn():
        return tf.zeros(tf.shape(result))

    result = tf.cond(is_finite, true_fn=true_fn, false_fn=false_fn)
    return result

##---------------##
## Training loop ##
##---------------##

totaller=0
n_episodes = 500
n_steps = int(100)
episode_rewards = np.zeros(n_episodes)
episode_length = np.zeros(n_episodes)
for episode in range(n_episodes):

    states = env.reset()

    agentQueus = {}

    for agentKey in A2C_agents.keys():
        state = states[agentKey]["curr_obs"]
        queue = deque()
        queue.extendleft([state])
        agentQueus[agentKey] = queue

    cum_rew = 0.
    stepDone = False

    for step in range(n_steps):
        # env.render(f"images/{str(episode).zfill(10)}_{str(step).zfill(10)}.png")
        curr_actions = {}
        for agentKey, agentObject in A2C_agents.items():
            queue = agentQueus[agentKey]
            allstates = tf.concat([x for x in queue], 0)
            allstates = np.reshape(allstates, [1, 1, 15, 15, 3])
            action = agentObject.get_action(allstates)
            curr_actions[agentKey] = action
        next_states, rewards, dones, _info = env.step(curr_actions) # agent id
        
########## CALCULATE CAUSAL INFLUENCE ##########
        agent_j = A2C_agents["agent-0"] 
        agent_k = A2C_agents["agent-1"] 

        # Left part of the input of kldiv
        agent_k_queue = agentQueus["agent-1"]
        agent_k_obs = tf.concat([x for x in agent_k_queue], 0)
        agent_k_obs = np.reshape(agent_k_obs, [1, 1, 15, 15, 3])
        A = agent_k.get_action(agent_k_obs) # Get action of the free agent
        A_obs, rewards, dones, _info = env.dummy_step({"agent-1": A})
        A_obs_READY = np.reshape(A_obs["agent-0"]["curr_obs"], [1, 1, 15, 15, 3])
        C = agent_j.get_action_probabilities(A_obs_READY) # actions van J wetende dat K action A heeft gedaan.
        # Right part of the input of kldiv
        agent_j_queue = agentQueus["agent-0"]
        agent_j_obs = tf.concat([x for x in agent_j_queue], 0)
        agent_j_obs = np.reshape(agent_j_obs, [1, 1, 15, 15, 3])
        B = agent_j.get_action_probabilities(agent_j_obs) # Action probs van j agent
        
        causal_reward = kl_div(C, B)

        ##########
        for agentKey, agentObject in A2C_agents.items():
            next_state = next_states[agentKey]["curr_obs"]

            reward = rewards[agentKey]
            if reward == 1:
                print("{} has {} in step {}".format(agentKey, reward, step))
            cum_rew += reward

            
            done = dones[agentKey]
            action = curr_actions[agentKey]

            queue = agentQueus[agentKey]
            prev_states = tf.concat([x for x in queue], 0)
            prev_states = np.reshape(prev_states, [1, 1, 15, 15, 3])
            queue.pop()
            queue.extendleft([next_state])
            allstates = tf.concat([x for x in queue], 0)
            allstates = np.reshape(allstates, [1, 1, 15, 15, 3])
            # not sure what good values for alpha and beta would be yet
            alpha = 0.5
            beta = 0.5
            if (agentKey == "Agent-1"):
                reward = (alpha * reward) + (beta * causal_reward)

            agentObject.train_model(prev_states, action, reward, allstates, done)

    episode_length[episode] = step
    episode_rewards[episode] = cum_rew

    print(f"[{episode}] Episode rewards: {cum_rew} after {step+1} steps")
    # for agentKey, agentObject in A2C_agents.items():
    #     agentObject.save()