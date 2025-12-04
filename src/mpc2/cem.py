from typing import List, Tuple

import click
from scipy.optimize import minimize

from mpc2.planner import ModelPredictivePlanner
import mpc2.utils as U
from mpc2.world_model import SimpleDiscreteTimeDynamics, WorldModel


class CrossEntropyMethodPlanner(ModelPredictivePlanner):
    """Simple sampling-based MPC planner using Cross-Entropy Method."""

    def __init__(
        self,
        state_cost: List[float],
        action_cost: List[float],
        control_bounds: Tuple[float, float],
        n_iterations: int,
        n_rollouts: int,
        n_elites: int,
        world_model: WorldModel,
        dt: float,
        **kwargs,
    ):
        super().__init__(world_model=world_model, dt=dt, **kwargs)
        self.Q = U.get_tensor_library().diag(state_cost)
        self.R = U.get_tensor_library().diag(action_cost)
        self.mu = U.get_tensor_library().zeros(2)
        self.sigma = U.get_tensor_library().ones(2)
        self.control_bounds = control_bounds
        self.n_rollouts = n_rollouts
        self.n_elites = n_elites
        self.n_iterations = n_iterations

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

        def rollout():
            actions = U.get_tensor_library().random.normal(loc=self.mu, scale=self.sigma, size=(self.horizon, 2))
            actions = U.get_tensor_library().clip(actions, self.control_bounds[0], self.control_bounds[1])
            return actions

        trajectories = []
        for _ in range(self.n_iterations):
            trajectories = []
            for _ in range(self.n_rollouts):
                traj = rollout()
                cost = self.cost_function(actions, initial_state, goal)
                trajectories.append((traj, cost))
            trajectories.sort(key=lambda x: x[1], reverse=True)
            self.mu = U.get_tensor_library().mean([traj for traj, _ in trajectories[:self.n_elites]], axis=0)
            self.sigma = U.get_tensor_library().std([traj for traj, _ in trajectories[:self.n_elites]], axis=0)

        return rollout()


@click.command()
@click.option("--n-cem-steps", type=int, default=100)
@click.option("--n-rollouts-per-step", type=int, default=32)
@click.option("--n-elites", type=int, default=8)
@click.option("--max-iterations", type=int, default=20)
@click.option("--horizon", type=int, default=5)
@click.option("--dt", type=float, default=1.0)
def run(n_cem_steps: int, n_rollouts_per_step: int, n_elites: int, max_iterations: int, horizon: int, dt: float):
    import numpy as np

    U.set_tensor_type(np.zeros((1,)))

    world_model = SimpleDiscreteTimeDynamics(dt=dt)
    planner = CrossEntropyMethodPlanner(
        world_model=world_model,
        state_cost=[10.0, 10.0],
        action_cost=[1.0, 1.0],
        control_bounds=(-2.0, 2.0),
        n_iterations=n_cem_steps,
        n_rollouts=n_rollouts_per_step,
        n_elites=n_elites,
        horizon=horizon,
        dt=dt,
    )
    state = {
        "position": U.make_tensor([0.0, 0.0]),
        "obstacles": U.make_tensor([[1.0, 1.0]]),
    }

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
