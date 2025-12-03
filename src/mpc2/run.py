from typing import Tuple
import queue

import click
import numpy as np

from mpc2.robot import Robot
from mpc2.server import Server
from mpc2.slsqp import SLSQPPlanner
import mpc2.utils as U
from mpc2.world_model import SimpleDiscreteTimeDynamics


@click.command()
@click.option("--horizon", type=int, default=5)
@click.option("--dt", type=float, default=1.0)
@click.option("--max-steps", type=int, default=100)
@click.option("--goal", type=tuple, default=[20.0, 35.0])
@click.option("--robot-initial-pos", type=tuple, default=(0.0, 0.0))
def main(
    horizon: int,
    dt: float,
    max_steps: int,
    goal: Tuple[float, float],
    robot_initial_pos: Tuple[float],
):
    U.set_tensor_type(np.zeros(1))

    dynamics = SimpleDiscreteTimeDynamics(dt=dt)
    initial_state = {
        "position": U.make_tensor(robot_initial_pos),
        "obstacles": U.make_tensor([[1.0, 1.0]]),
    }

    robot = Robot(initial_state, dynamics)

    planner = SLSQPPlanner(
        world_model=dynamics,
        state_cost=[10.0, 10.0],
        action_cost=[1.0, 1.0],
        control_bounds=(-2.0, 2.0),
        horizon=horizon,
        dt=dt,
    )

    server = Server({"obstacles": U.make_tensor([[np.inf, np.inf]])}, planner)

    goal = U.make_tensor(goal)

    actions = queue.Queue()

    while robot.t <= max_steps and U.tensor_norm(robot.pos - goal) > 0.5:
        should_update = (obs := robot.get_observations()) is not None
        if should_update:
            print(f"Updating server at t={robot.t:.2f}")
            server.update(robot.pos, obs, robot.t)

        if actions.empty() or should_update:
            for control in server.plan(goal):
                actions.put(control)
        control = actions.get()
        print(
            f"[t={robot.t:.2f}]: pos=({robot.pos[0]:.2f}, {robot.pos[1]:.2f}), control=({control[0]:.2f}, {control[1]:.2f})"
        )
        robot.step(control)

    if U.tensor_norm(robot.pos - goal) <= 0.5:
        print(
            f"Goal reached in t={robot.t:.2f}. robot_pos=({robot.pos[0]:.2f}, {robot.pos[1]:.2f})"
        )
    else:
        print(
            f"Failed to reach goal in t={robot.t:.2f}. robot_pos=({robot.pos[0]:.2f}, {robot.pos[1]:.2f})"
        )


if __name__ == "__main__":
    main()
