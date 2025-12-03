import queue
import time
from typing import Tuple

import click
import crypten
import numpy as np
import torch

from .planner import ModelPredictivePlanner


class NavigationPlanner(ModelPredictivePlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plan(self, initial_state, goal):
        pass

class Server:
    def __init__(self, initial_pos: Tuple[float, float]):
        self._initial_pos = initial_pos
        self._obs_buffer = queue.Queue(maxsize=10)
        self.reset()
    
    def reset(self):
        self.pos = torch.tensor(self._initial_pos)
        self.obstacles = torch.zeros(1, 3)
        self._obs_buffer.queue.clear()

    def world_model(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, pos: Tuple[float, float], obs: torch.Tensor):
        self.pos = torch.tensor(pos)
        if self._obs_buffer.full():
            n_expired = self._obs_buffer.get()
            self.state = self.state[n_expired:]
        self._obs_buffer.put(obs)
        self.state = torch.cat([self.state, obs], dim=0)

    def plan(self, goal: torch.Tensor):
        pass

@click.command()
def run():
    server = Server()
    while True:
        obs = np.random.rand(1, 3)
        server.update(obs)
        print(server.state)
        time.sleep(1)

if __name__ == "__main__":
    run()