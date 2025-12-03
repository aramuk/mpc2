import abc

from .types import State, Tensor
from .utils import tensor_norm

class WorldModel(abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, state: State, action: Tensor) -> State:
        raise NotImplementedError


class SimpleDiscreteTimeDynamics(WorldModel):
    def __init__(self, dt: float):
        super().__init__()
        self.dt = dt

    def __call__(self, state: State, action: Tensor) -> State:
        pos = state.get("position")
        obstacles = state.get("obstacles")
        if obstacles is not None:
            for obstacle in obstacles:
                if tensor_norm(pos - obstacle) < 1.0:
                    return pos
        return pos + action * self.dt
