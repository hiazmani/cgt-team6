import numpy as np
from numpy.random import rand

from social_dilemmas.envs.agent import TrappedAgent
from social_dilemmas.envs.agent import FreeAgent

from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import TRAPPED_BOX_MAP, TRAPPED_BOX_MAP2

APPLE_RADIUS = 2

# Add custom actions to the agent
_TRAPPED_BOX_ACTIONS = {"FREE": [0, 0]}  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

TRAPPED_BOX_VIEW_SIZE = 7


class TrappedBoxEnv(MapEnv):
    def __init__(
        self,
        ascii_map=TRAPPED_BOX_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,                   
    ):
        self.n_actions = 6
        super().__init__(
            ascii_map,
            _TRAPPED_BOX_ACTIONS,
            TRAPPED_BOX_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(8, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        agent_id = "agent-" + str(0)
        points = self.spawn_points
        
        spawn_point = points[0]
        
        rotation = self.spawn_rotation()
        grid = map_with_agents
        agent = TrappedAgent(agent_id, spawn_point, rotation, grid, view_len=TRAPPED_BOX_VIEW_SIZE)
        self.agents[agent_id] = agent

        agent_id = "agent-" + str(1)
        spawn_point = points[1]
        
        rotation = self.spawn_rotation()
        grid = map_with_agents
        agent = FreeAgent(agent_id, spawn_point, rotation, grid, view_len=TRAPPED_BOX_VIEW_SIZE)
        self.agents[agent_id] = agent

    

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")

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
        updates = []
        if isinstance(agent, FreeAgent):  
            agent.free_agent()
            updates = [
                [2, 10, b" "], [2,11,b" "], [2, 12,b" "],
                [3, 10, b" "],              [3, 12,b" "],
                [4, 10, b" "], [4,11,b" "], [4, 12,b" "]
            ] 
        
        
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":

                spawn_prob = 0.25
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples
