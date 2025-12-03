from typing import List, Tuple

import click
from scipy.optimize import minimize

from mpc2.planner import ModelPredictivePlanner
import mpc2.utils as U
from mpc2.world_model import SimpleDiscreteTimeDynamics, WorldModel


class SLSQPPlanner(ModelPredictivePlanner):
    """Incredibly simple MPC planner for navigation in a 2D grid."""

    def __init__(
        self,
        state_cost: List[float],
        action_cost: List[float],
        control_bounds: Tuple[float, float],
        world_model: WorldModel,
        dt: float,
        **kwargs,
    ):
        super().__init__(world_model=world_model, dt=dt, **kwargs)
        self.Q = U.get_tensor_library().diag(state_cost)
        self.R = U.get_tensor_library().diag(action_cost)
        self.control_bounds = control_bounds

    def cost_function(self, actions, initial_state, goal):
        """Total cost over horizon"""
        u = actions.reshape((self.horizon, 2))
        cost = 0.0
        x = initial_state.copy()
        obstacles = initial_state.get("obstacles")

        for t in range(self.horizon):
            # Predict next state
            x = self.model(x, u[t])

            # Obstacle cost (penalize collisions)
            if obstacles is not None:
                for obstacle in obstacles:
                    if U.tensor_norm(x["position"] - obstacle) < 1.0:
                        cost += 1e10
                        return cost

            # State cost (distance to goal)
            state_error = x["position"] - goal
            cost += state_error.T @ self.Q @ state_error

            # Control cost (penalize large velocities)
            cost += u[t].T @ self.R @ u[t]

        return cost

    def plan(self, initial_state, goal):
        actions = U.get_tensor_library().zeros(self.horizon * 2)
        bounds = [self.control_bounds] * (self.horizon * 2)

        result = minimize(
            self.cost_function,
            actions,
            args=(initial_state, goal),
            method="SLSQP",
            bounds=bounds,
        )
        # Return full trajectory
        trajectory = result.x.reshape((self.horizon, 2))
        return trajectory


@click.command()
@click.option("--max-iterations", type=int, default=20)
@click.option("--horizon", type=int, default=5)
@click.option("--dt", type=float, default=1.0)
def run(max_iterations: int, horizon: int, dt: float):
    import numpy as np
    U.set_tensor_type(np.zeros((1,)))

    world_model = SimpleDiscreteTimeDynamics(dt=dt)
    planner = SLSQPPlanner(
        world_model=world_model,
        state_cost=[10.0, 10.0],
        action_cost=[1.0, 1.0],
        control_bounds=(-2.0, 2.0),
        horizon=horizon,
        dt=dt,
    )
    state = {"position": U.make_tensor([0.0, 0.0]), "obstacles": U.make_tensor([[1.0, 1.0]])}

    goal = U.make_tensor([20.0, 35.0])
    actions = planner.plan(state, goal)
    print(actions)

    trajectory = [state["position"].copy()]

    # Simulate navigation
    for step in range(max_iterations):
        # Compute optimal control
        control = planner.plan(state, goal)[0]

        # Apply control and update position
        state = planner.model(state, control)
        trajectory.append(state["position"].copy())
        pos = state["position"]

        print(
            f"Step {step}: pos=({pos[0]:.2f}, {pos[1]:.2f}), control=({control[0]:.2f}, {control[1]:.2f})"
        )

        # Check if reached goal
        if U.tensor_norm(pos - goal) < 0.5:
            print("Goal reached!")
            break

    print("\nTrajectory:")
    for i, p in enumerate(trajectory):
        print(f"  {i}: ({p[0]:.2f}, {p[1]:.2f})")


if __name__ == "__main__":
    run()
