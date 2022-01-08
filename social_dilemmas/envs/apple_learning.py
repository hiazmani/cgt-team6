from social_dilemmas.envs import agent
from social_dilemmas.envs.agent import FreeAgent
from social_dilemmas.maps import TRAPPED_BOX_MAP2

from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.envs.agent import AppleAgent
from numpy.random import rand

import numpy as np

_APPLE_LEARNING_AGENT_ACTS = {}

APPLE_LEARNING_VIEW_SIZE = 7

class AppleLearningEnv(MapEnv):
    def __init__(
        self,
        random_spawn_agent,
        random_spawn_apple,
        ascii_map=TRAPPED_BOX_MAP2,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,                   
    ):
        self.random_spawn_agent = random_spawn_agent
        self.random_spawn_apple = random_spawn_apple
        self.n_actions = 5
        super().__init__(
            ascii_map,
            _APPLE_LEARNING_AGENT_ACTS,
            APPLE_LEARNING_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.apple_points.append([row, col])
                # if self.base_map[row, col] == b"B":
                #     self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(5, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        # Create a single agent one one of the spawn points
        agent_id = "agent-" + str(0)
        points = self.spawn_points
        # spawn_point = None
        if self.random_spawn_agent:
            random = np.random.randint(len(self.spawn_points))
            spawn_point = points[random]
        else:
            spawn_point = points[0]
        
        rotation = self.spawn_rotation()
        grid = map_with_agents
        agent = AppleAgent(agent_id, spawn_point, rotation, grid, view_len=APPLE_LEARNING_VIEW_SIZE)
        self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        # Retrieve the agent's position
        agent_positions = self.agent_pos
        apple_spawn_point = None
        # If we are using random starting positions, assign a random position to the apple (that is not the player's position)
        if self.random_spawn_apple:
            random = np.random.randint(len(self.apple_points))
            pos = self.apple_points[random]
            while pos in agent_positions:
                random = np.random.randint(len(self.apple_points))
                pos = self.apple_points[random]
            apple_spawn_point = pos
        else:
            # If agent is currently standing on the location where the default apple should spawn wait for the agent to move
            if self.apple_points[-1] in agent_positions:
                return
            apple_spawn_point = self.apple_points[-1]
        self.single_update_map(apple_spawn_point[0], apple_spawn_point[1], b"A")

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        # No custom action in the apple-learning environment        
        return []

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        # Update the map with the new apples
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
        # Points that will contain new apples
        new_apple_points = []

        # Retrieve the agent's current position
        agent_positions = self.agent_pos

        # Check whether there already is an apple somewhere
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # Check whether this location contains an apple
            if self.world_map[row, col] == b"A":
                # If there already is an apple, return we do not want to add an additional apple
                return new_apple_points
        # Place an apple
        for i in range(len(self.apple_points)):
            if self.random_spawn_apple:
                random = np.random.randint(len(self.apple_points))
                row, col = self.apple_points[random]
            else:
                row, col = self.apple_points[-1]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions:
                new_apple_points.append((row, col, b"A"))
                return new_apple_points
            # If agent is currently standing on the location where the default apple should spawn wait for the agent to move
            elif not self.random_spawn_apple:
                return new_apple_points
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples