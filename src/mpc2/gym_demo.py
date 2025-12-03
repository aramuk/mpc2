import copy
import queue
from typing import List, Literal, Tuple

import click
import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


Action = Tuple[float, float, float]
State = List[np.ndarray]
Reward = float
Trajectory = List[Tuple[State, Action, Reward]]
SamplingStrategy = Literal["random_shooting", "cross_entropy_method"]


class MultistepEnv():
    def __init__(self, env_name: str, step_size: int, **kwargs) -> None:
        self.env = gym.make(env_name, **kwargs)
        self.step_size = step_size

    def step(self, action):
        obs, rewards, termination, truncation, infos = None, 0.0, False, False, None
        for i, _ in enumerate(range(self.step_size)):
            if i == 0:
                o, r, term, trunc, inf = self.env.step(action)
            else:
                o, r, term, trunc, inf = self.env.step(np.zeros_like(action))
            obs = o
            rewards += r
            termination |= term
            truncation |= trunc
            infos = inf
        return obs, rewards, termination, truncation, infos

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        self.env.close()

def mpc_planner(env: gym.Env, horizon: int = 60, n_trajectories: int = 4, sampling_strategy: SamplingStrategy = "random_shooting") -> Trajectory:
    """Plan a trajectory for the agent to follow using Model-Predictive Control."""

    def J(trajectory: Trajectory) -> float:
        return sum([reward for _, _, reward in trajectory])

    # We can use the enviroment itself as a perfect simulator / world-model.
    trajectories = []
    for _ in tqdm(range(n_trajectories), desc="Sampling Trajectories for MPC"):
        trajectory = []
        state = env.reset()
        for _ in range(horizon):
            if sampling_strategy == "random_shooting":
                action = env.action_space.sample()
            else:
                action = None
            next_state, reward, _,_,_ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
        trajectories.append(trajectory)

    return min(trajectories, key=J)


@click.command()
@click.option("--visualize/--no-visualize", default=True)
def navigation_demo(visualize: bool):
    """Run a demonstration of an agent learning to complete the CarRacing-v3 environment."""

    metrics = {"obs": [], "action": [], "reward": [], "terminated": [], "truncated": []}
    env = MultistepEnv("CarRacing-v3", 60)
    env.reset()

    actions = queue.SimpleQueue()
    for _ in range(16):
        if actions.empty():
            for _, act, _ in mpc_planner(copy.deepcopy(env)):
                actions.put(act)
        action = actions.get()
        obs, reward, terminated, truncated, info = env.step(action)
        metrics["obs"].append(obs)
        metrics["action"].append(action)
        metrics["reward"].append(reward)
        metrics["terminated"].append(terminated)
        metrics["truncated"].append(truncated)
        if terminated or truncated:
            break
    env.close()

    if visualize:
        # Display video using metrics["obs"] as frames
        fig, ax = plt.subplots()
        ims = []
        for frame in metrics["obs"]:
            im = ax.imshow(frame, animated=True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=1000/15, blit=True, repeat_delay=50000, repeat=True)
        plt.show()

        fig, ax = plt.subplots(3, 1, figsize=(10, 7))
        ax[0].set_title("Control Inputs")
        ax[0].plot(metrics["action"])
        ax[0].plot([a[0] for a in metrics["action"]], label="Steering", color="b")
        ax[0].plot([a[1] for a in metrics["action"]], label="Gas", color="g")
        ax[0].plot([a[2] for a in metrics["action"]], label="Brake", color="r")
        ax[0].legend()
        ax[1].set_title("Reward")
        ax[1].plot(metrics["reward"], color="g")
        ax[2].set_title("Termination")
        ax[2].plot(metrics["terminated"], color="r", label="Terminated")
        ax[2].plot(metrics["truncated"], color="y", label="Truncated")
        ax[2].legend()
        plt.show()


if __name__ == "__main__":
    navigation_demo()