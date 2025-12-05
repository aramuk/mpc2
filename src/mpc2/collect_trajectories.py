import json
from pathlib import Path
from typing import Optional

import click
import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from omnigibson.envs import DataCollectionWrapper
from pydantic import BaseModel, Field
from pydanclick import from_pydantic

FSAF_OBJECT_SCENES_PATH = Path("../../fsaf_object_scenes.json")


class CollectionConfig(BaseModel):
    # Scene configuration
    model_id: str
    category: Optional[str] = None
    scene_id: Optional[str] = None
    scene_index: int = 0
    # Robot configuration
    robot_type: str = "Freight"
    # Env configuration
    action_step_hz: int = 10
    physics_step_hz: int = 120
    # DataCollectionWrapper args
    only_successes: bool = False
    output_path: Path = Field(default=Path.cwd() / "fsaf-logs" / "robot_trajectories" / "")
    num_trajectories: int = 4096


def create_og_config(cfg: CollectionConfig):
    if cfg.scene_id is None:
        with FSAF_OBJECT_SCENES_PATH.open("r") as f:
            model_id2scenes = json.load(f)
        if len(scenes := model_id2scenes.get(cfg.obj_model_id, [])) > 0:
            cfg.scene_id = scenes[cfg.scene_index]

    # If scene is not provided and object is not found in any scenes,
    # create an empty scene with only the object.
    if cfg.scene_id is None:
        return {
            "scene": {"type": "Scene", "floor_plane_visible": False},
            "objects": {
                "type": "DatasetObject",
                "model_id": cfg.model_id,
                "category": cfg.category,
            },
            "robot": {
                "type": "Freight",
                "obs_modalities": ["rgb"],
                "action_type": "continuous",
                "action_normalize": True,
            },
            "env": {"action_timestep": 1.0 / 10.0, "physics_timestep": 1 / 120.0},
            "render": {"viewer_width": 1024, "viewer_height": 1024},
        }
    else:
        scene_config = {"type": "InteractiveTraversableScene", "scene_model": cfg.scene_id}

    # Add the robot we want to load
    robot0_config = {
        "type": cfg.robot_type,
        "obs_modalities": ["rgb"],
        "action_type": "continuous",
        "action_normalize": True,
    }

    # Compile config
    config = {
        "scene": scene_config,
        "robots": [robot0_config],
        "env": {"action_timestep": 1.0 / cfg.action_step_hz, "physics_timestep": 1.0 / cfg.physics_step_hz},
        "render": {"viewer_width": 1024, "viewer_height": 1024},
    }
    return config


@click.command
@from_pydantic(CollectionConfig)
def main(args: CollectionConfig):
    # Create the environment
    env = og.Environment(configs=create_og_config(args))
    robot = env.robots[0]
    # TODO:
    # - House scenes and int scenes are stable. Everything else, just use empty scene.

    # load IK controller
    controller_config = {
        f"arm_{robot.default_arm}": {"name": "InverseKinematicsController", "mode": "pose_absolute_ori"}
    }
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    action_primitives = StarterSemanticActionPrimitives(env, robot, skip_curobo_initilization=True)

    # Warm-up DLSS anti-aliasing neural net.
    og.sim.step()
    for _ in range(10):
        og.sim.render()

    # Wrap it with DataCollectionWrapper
    wrapped_env = DataCollectionWrapper(
        env=env,
        output_path=args.output_path,
        only_successes=args.only_successes,  # Set to True to only save successful episodes
    )

    # TODO: tasks
    # - Toggle a button
    # - Cook a steak on a heatsource
    # - Fill a cup with water
    # - Maybe: fill a sink

    # Use the wrapped environment as you would normally
    obs, info = wrapped_env.reset()
    for _ in range(args.num_trajectories):
        obj = env.scene.object_registry("name", f"{args.model_id}0")
        for action in action_primitives._toggle_on(obj):
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
        wrapped_env.reset()
    # Save the collected data
    wrapped_env.save_data()

    # TODO: dump point cloud heatmap+estimates into .ply file using trimesh


if __name__ == "__main__":
    main()
