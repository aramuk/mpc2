import abc
from typing import Tuple

from .types import State, Tensor, Trajectory
from .world_model import WorldModel

class ModelPredictivePlanner(abc.ABC):
    def __init__(self, model: WorldModel, horizon: int, dt: float):
        super().__init__()
        self.horizon = horizon
        self.dt = dt
        self.model = model

    @abc.abstractmethod
    def plan(self, initial_state: State, goal: Tensor) -> Trajectory:
        raise NotImplementedError