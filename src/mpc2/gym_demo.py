from typing import List, Tuple

import click
import gymnasium as gym
import matplotlib.pyplot as plt

def mpc_planner(env: gym.Env) -> List[Tuple[float, float, float]]:
    """Plan a trajectory for the agent to follow using Model-Predictive Control."""

    # We can use the enviroment itself as a perfect simulator / world-model.


    return 


@click.command()
@click.option("--visualize/--no-visualize", default=True)
def navigation_demo(visualize: bool):
    """Run a demonstration of an agent learning to complete the CarRacing-v3 environment."""

    metrics = {"obs": [], "action": [], "reward": [], "terminated": [], "truncated": []}
    env = gym.make("CarRacing-v3", render_mode="human" if visualize else None)
    env.reset()
    while True:
        action = env.action_space.sample()
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