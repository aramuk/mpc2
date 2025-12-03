import abc

from mpc2.annotations import State, TensorType, Trajectory
from mpc2.world_model import WorldModel

class ModelPredictivePlanner(abc.ABC):
    def __init__(self, world_model: WorldModel, horizon: int, dt: float):
        super().__init__()
        self.horizon = horizon
        self.dt = dt
        self.model = world_model

    @abc.abstractmethod
    def plan(self, initial_state: State, goal: TensorType) -> Trajectory:
        raise NotImplementedError

    @abc.abstractmethod
    def cost_function(self, actions: TensorType, initial_state: TensorType, goal: TensorType) -> TensorType:
        raise NotImplementedError