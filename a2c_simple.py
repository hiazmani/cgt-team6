from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
import numpy as np
import random as random
import tensorflow as tf

##---------------------------------------------------------------------------------------------------------------##
## Code inspired by the following tutorials:                                                                     ##
##  - https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py     ##
##  - https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb ##
##---------------------------------------------------------------------------------------------------------------##

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, observation_size, action_size, learning_rates, fc_args, gamma, saved_weights_dir = None):
        # get size of state and action
        self.observation_size = observation_size
        self.action_size = action_size
        self.value_size = 1

        # Extract the parameters for the fully-connected layers
        self.n_dense_layer_neurons = fc_args[0] # Number of neurons in each of the fully connected layers
        self.dense_activation_function = fc_args[1]

        # Extract the learning rates
        self.actor_lr = learning_rates[0]
        self.critic_lr = learning_rates[1]

        # Save the discount factor
        self.discount_factor = gamma

        # create model for policy network
        self.actor = self.build_actor_nn()
        self.critic = self.build_critic_nn()

    def build_actor_nn(self):
        """
            Creates the Actor neural network: outputs a probability distribution of the actions given a specific input state (observation)
        """
        # Attach all the layers of our neural network sequentially
        actor = Sequential()
        # Use 2 hidden fully connected layers
        actor.add(Dense(self.n_dense_layer_neurons, input_dim=self.observation_size, activation=self.dense_activation_function, kernel_initializer='he_uniform'))
        # Use a softmax output layer with action_size output neurons to produce a probability distribution of the actions 
        actor.add(Dense(self.action_size, activation="softmax"))
        # Use the categorical cross entropy loss function
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.actor_lr))
        # Return the newly created Actor NN
        return actor

    # critic: state is input and value of state is output of model
    def build_critic_nn(self):
        """
            Creates the Actor neural network: outputs a probability distribution of the actions given a specific input state (observation)
        """
        # Attach all the layers of our neural network sequentially
        critic = Sequential()
        # Use 2 hidden fully connected layers
        critic.add(Dense(self.n_dense_layer_neurons, input_dim=self.observation_size, activation=self.dense_activation_function, kernel_initializer='he_uniform'))
        # Use a softmax output layer to produce the critic value
        critic.add(Dense(self.value_size, activation="softmax"))
        # Use the mean squared loss error function
        critic.compile(loss="mse", optimizer=Adam(learning_rate=self.critic_lr))
        # Return the newly created Critic NN
        return critic

    # Predict the policy (probability of each action) and pick action based on probabilities
    def get_action(self, state):
        policy = self.actor.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # Predict the policy (probability of each action) and return the probabilities
    def get_action_probabilities(self, state):
        return self.actor.predict(state).flatten()

    def compute_advantage(self, action, reward, value, next_value, done):
        advantages = np.zeros((1, self.action_size))
        if done:
            advantages[0][action] = reward - value
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
        return advantages

    def compute_target(self, reward, next_value, done):
        target = np.zeros((1, self.value_size))
        if done:
            target[0][0] = reward
        else:
            target[0][0] = reward + self.discount_factor * next_value
        return target

    # Update the model
    def train_model(self, state, action, reward, next_state, done):
        # Compute the Critic value for the current observation
        value = self.critic.predict(state)[0]
        # Compute the Critic value for the next observation
        next_value = self.critic.predict(next_state)[0]

        # Compute the advantages for the Actor
        advantages = self.compute_advantage(action, reward, value, next_value, done)
        # Compute the target for the Critic
        target = self.compute_target(reward, next_value, done)

        # Fit the actor using the computed advantages
        self.actor.fit(state, advantages, epochs=1, verbose=0)
        # Fit the critic using the computed target
        self.critic.fit(state, target, epochs=1, verbose=0)


    def save_weights(self, model_dir):
        # Use Tensorflow's built-in saving and loading of models to save the weights of the Actor and Critic
        self.critic.save_weights(f"./{model_dir}/a2c_critic.h5")
        self.actor.save_weights(f"./{model_dir}/a2c_actor.h5")
    
    def load_weights(self, model_dir):
        # Use Tensorflow's built-in saving and loading of models to load-in the weights of the Actor and Critic
        self.critic.load_weights(f"./{model_dir}/a2c_critic.h5")
        self.actor.load_weights(f"./{model_dir}/a2c_actor.h5")