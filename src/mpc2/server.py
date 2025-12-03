import queue

from mpc2.annotations import State, TensorType
from mpc2.planner import ModelPredictivePlanner
from mpc2.world_model import WorldModel
import mpc2.utils as U


class Server:
    def __init__(
        self,
        initial_state: State,
        planner: ModelPredictivePlanner,
    ):
        self._initial_state = initial_state
        self._obs_buffer = queue.Queue(maxsize=10)
        self.reset()
        self.planner = planner
        self.t = 0.0

    def reset(self):
        self.t = 0.0
        self.state = self._initial_state.copy()
        self._obs_buffer.queue.clear()
        self._obs_buffer.put(self.state["obstacles"].shape[0])

    @property
    def pos(self) -> State:
        return self.state["position"]

    def update(self, pos: TensorType, obs: TensorType, ts: float = 1.0):
        if self._obs_buffer.full():
            n_expired = self._obs_buffer.get()
            self.state["obstacles"] = self.state["obstacles"][n_expired:]
        self._obs_buffer.put(obs.shape[0])
        self.state["obstacles"] = U.tensor_cat([self.state["obstacles"], obs])
        self.state["position"] = pos
        self.t = ts

    def plan(self, goal: TensorType):
        return self.planner.plan(self.state, goal)
