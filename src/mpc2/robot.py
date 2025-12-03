import numpy as np

from mpc2.annotations import State, TensorType
from mpc2.world_model import WorldModel

class Robot:
    def __init__(self, initial_state: State, dynamics: WorldModel):
        self._initial_state = initial_state
        self.dynamics = dynamics
        self.reset()

    def reset(self):
        self.state = self._initial_state.copy()
        self.t = 0.0

    @property
    def pos(self):
        return self.state["position"]

    def step(self, action: TensorType):
        self.state = self.dynamics(self.state, action)
        self.t += 1.0

    def get_observations(self):
        if int(self.t) % 10 == 0:
            return self.state["position"] + np.ones((1, 2))
        else:
            return None
