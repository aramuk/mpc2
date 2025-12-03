import abc

from mpc2.annotations import State, TensorType
import mpc2.utils as U

class WorldModel(abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, state: State, action: TensorType) -> State:
        raise NotImplementedError


class SimpleDiscreteTimeDynamics(WorldModel):
    def __init__(self, dt: float):
        super().__init__()
        self.dt = dt

    def __call__(self, state: State, action: TensorType) -> State:

        pos = state.get("position")
        obstacles = state.get("obstacles")
        if obstacles is not None:
            for obstacle in obstacles:
                if U.tensor_norm(pos - obstacle) < 1.0:
                    return {"position": pos, "obstacles": state.get("obstacles")}
        
        return {"position": pos + action * self.dt, "obstacles": state.get("obstacles")}
