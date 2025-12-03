import queue
import time
from typing import List, Tuple

import click

from mpc2.annotations import State, TensorType
from mpc2.slsqp import SLSQPPlanner
from mpc2.planner import ModelPredictivePlanner
from mpc2.world_model import SimpleDiscreteTimeDynamics, WorldModel
import mpc2.utils as U


class Server:
    def __init__(
        self,
        initial_state: State,
        world_model: WorldModel,
        planner: ModelPredictivePlanner,
    ):
        self._initial_state = initial_state
        self._obs_buffer = queue.Queue(maxsize=10)
        self.reset()
        self.world_model = world_model
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


if __name__ == "__main__":
    import numpy as np

    U.set_tensor_type(np.zeros(1))

    world_model = SimpleDiscreteTimeDynamics(dt=1.0)
    planner = SLSQPPlanner(
        world_model=world_model,
        state_cost=[10.0, 10.0],
        action_cost=[1.0, 1.0],
        control_bounds=(-2.0, 2.0),
        horizon=5,
        dt=1.0,
    )
    state = {
        "position": U.make_tensor([0.0, 0.0]),
        "obstacles": U.make_tensor([[1.0, 1.0]]),
    }

    server = Server(state, world_model, planner)

    goal = U.make_tensor([20.0, 35.0])

    actions = queue.Queue()
    robot_pos = server.pos
    robot_t = 0.0
    while server.t <= 100.0 and U.tensor_norm(robot_pos - goal) > 0.5:
        should_update = np.random.random() < 0.2
        if should_update:
            print(f"Updating server at t={robot_t:.2f}")
            server.update(robot_pos, robot_pos + np.ones((1, 2)), robot_t)

        if actions.empty() or should_update:
            for control in server.plan(goal):
                actions.put(control)
        control = actions.get()
        print(
            f"[t={robot_t:.2f}]: pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), control=({control[0]:.2f}, {control[1]:.2f})"
        )
        robot_pos = server.planner.model(
            {"position": robot_pos, "obstacles": None}, control
        )["position"]
        robot_t += 1.0
    if U.tensor_norm(robot_pos - goal) <= 0.5:
        print(f"Goal reached in t={robot_t:.2f}. robot_pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
    else:
        print(f"Failed to reach goal in t={robot_t:.2f}. robot_pos=({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")