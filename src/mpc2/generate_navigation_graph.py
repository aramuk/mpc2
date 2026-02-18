"""
Generate a navigation graph by sensor-driven exploration of a Habitat-sim scene.

A robot is placed in a scene and uses its depth sensor to detect open space,
dropping waypoint nodes as it travels and connecting them with edges.  Three
exploration modes (frontier-chase, backtrack, random-walk) ensure broad
coverage.

Run with the ``habitat`` conda environment::

    python -m mpc2.generate_navigation_graph --scene-name apartment_1
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import magnum as mn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import habitat_sim
from habitat_sim.utils import common as utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scene registry
# ---------------------------------------------------------------------------

SCENE_REGISTRY: Dict[str, Dict[str, str]] = {
    "apartment_1": {
        "scene": "scene_datasets/habitat-test-scenes/apartment_1.glb",
    },
    "skokloster-castle": {
        "scene": "scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    },
    "van-gogh-room": {
        "scene": "scene_datasets/habitat-test-scenes/van-gogh-room.glb",
    },
    "mp3d_example": {
        "scene": "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",
        "dataset_config": "scene_datasets/mp3d_example/mp3d.scene_dataset_config.json",
    },
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExplorationConfig:
    scene_name: str = "apartment_1"
    data_path: str = "data/habitat/"
    sensor_height: float = 1.5
    sensor_resolution: int = 256
    forward_amount: float = 0.25
    turn_amount: float = 10.0
    max_steps: int = 2000
    waypoint_spacing: float = 0.5
    revisit_radius: float = 0.3
    stall_limit: int = 200
    seed: int = 42
    depth_fov_deg: float = 90.0
    free_space_threshold: float = 1.5
    obstacle_threshold: float = 0.3
    output_dir: str = "logs/nav_graphs/"


# ---------------------------------------------------------------------------
# Step 2 – Simulator setup
# ---------------------------------------------------------------------------


def make_sim_config(cfg: ExplorationConfig) -> habitat_sim.Configuration:
    """Build a ``habitat_sim.Configuration`` with RGB + depth sensors."""
    scene_entry = SCENE_REGISTRY[cfg.scene_name]
    data = Path(cfg.data_path)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = str(data / scene_entry["scene"])
    if "dataset_config" in scene_entry:
        sim_cfg.scene_dataset_config_file = str(
            data / scene_entry["dataset_config"]
        )
    sim_cfg.enable_physics = False

    res = cfg.sensor_resolution
    h = cfg.sensor_height

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [res, res]
    color_spec.position = [0.0, h, 0.0]
    color_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [res, res]
    depth_spec.position = [0.0, h, 0.0]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=cfg.forward_amount),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=cfg.turn_amount),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=cfg.turn_amount),
        ),
        "turn_left_small": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=5.0),
        ),
        "turn_right_small": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=5.0),
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def create_simulator(
    cfg: ExplorationConfig,
) -> Tuple[habitat_sim.Simulator, habitat_sim.Agent]:
    """Create, seed, and return ``(sim, agent)`` placed at a random navigable point."""
    hab_cfg = make_sim_config(cfg)
    sim = habitat_sim.Simulator(hab_cfg)
    sim.seed(cfg.seed)

    if not sim.pathfinder.is_loaded:
        raise RuntimeError(
            f"NavMesh not loaded for scene '{cfg.scene_name}'. "
            "Ensure a .navmesh file accompanies the .glb scene."
        )

    agent = sim.initialize_agent(0)
    sim.pathfinder.seed(cfg.seed)
    start = sim.pathfinder.get_random_navigable_point()
    state = habitat_sim.AgentState()
    state.position = start
    agent.set_state(state)
    return sim, agent


# ---------------------------------------------------------------------------
# Step 3 – Depth image analysis
# ---------------------------------------------------------------------------

N_SECTORS = 7


def analyze_depth_image(
    depth_obs: np.ndarray, cfg: ExplorationConfig
) -> Dict[str, Any]:
    """Return sector scores, best direction, obstacle flag, and open fraction."""
    H, W = depth_obs.shape

    # Vertical weighting: emphasise the centre rows (floor-level obstacles).
    vert_weights = np.exp(-0.5 * ((np.arange(H) - H / 2) / (H / 4)) ** 2)
    vert_weights /= vert_weights.sum()

    # Weighted column means
    col_means = vert_weights @ depth_obs  # shape (W,)

    # Split into sectors
    sector_edges = np.linspace(0, W, N_SECTORS + 1, dtype=int)
    sector_scores = np.zeros(N_SECTORS)
    for i in range(N_SECTORS):
        sector_scores[i] = col_means[sector_edges[i] : sector_edges[i + 1]].mean()

    best_sector = int(np.argmax(sector_scores))
    # Map sector index to [-1, 1]  (0 → -1, N-1 → +1)
    best_direction = 2.0 * best_sector / (N_SECTORS - 1) - 1.0

    # Centre sector obstacle check
    centre = N_SECTORS // 2
    obstacle_ahead = bool(sector_scores[centre] < cfg.obstacle_threshold)

    # Open fraction
    valid = np.isfinite(depth_obs) & (depth_obs > 0)
    if valid.any():
        open_fraction = float((depth_obs[valid] > cfg.free_space_threshold).mean())
    else:
        open_fraction = 0.0

    return {
        "sector_scores": sector_scores,
        "best_direction": best_direction,
        "obstacle_ahead": obstacle_ahead,
        "open_fraction": open_fraction,
    }


# ---------------------------------------------------------------------------
# Step 4 – Exploration memory / graph builder
# ---------------------------------------------------------------------------


class ExplorationState(enum.Enum):
    EXPLORE = "explore"
    BACKTRACK = "backtrack"
    RANDOM_WALK = "random_walk"


class ExplorationMemory:
    """Incrementally builds a navigation graph during exploration."""

    def __init__(self, cfg: ExplorationConfig):
        self.cfg = cfg
        self.graph: nx.Graph = nx.Graph()
        self.node_positions: Dict[int, np.ndarray] = {}
        self.visit_counts: Dict[int, int] = {}
        self.backtrack_stack: List[int] = []
        self._next_id = 0
        self.last_node_id: Optional[int] = None
        self.distance_since_last_node: float = 0.0
        self.steps_since_new_node: int = 0
        self._prev_position: Optional[np.ndarray] = None
        self._pathfinder: Optional[Any] = None  # set externally

    # -- helpers ----------------------------------------------------------

    def find_nearest_node(
        self, pos: np.ndarray
    ) -> Tuple[Optional[int], float]:
        if not self.node_positions:
            return None, float("inf")
        ids = list(self.node_positions.keys())
        pts = np.array([self.node_positions[i] for i in ids])
        dists = np.linalg.norm(pts - pos, axis=1)
        idx = int(np.argmin(dists))
        return ids[idx], float(dists[idx])

    def _add_node(self, pos: np.ndarray) -> int:
        nid = self._next_id
        self._next_id += 1
        self.graph.add_node(
            nid,
            pos=pos.copy(),
            pos_2d=(float(pos[0]), float(pos[2])),
            visit_count=1,
        )
        self.node_positions[nid] = pos.copy()
        self.visit_counts[nid] = 1
        return nid

    # -- main update ------------------------------------------------------

    def update(
        self, position: np.ndarray, depth_analysis: Dict[str, Any]
    ) -> Optional[int]:
        """Process new agent position; return new node id or ``None``."""
        pos = np.array(position, dtype=float)

        # Accumulate distance
        if self._prev_position is not None:
            self.distance_since_last_node += float(
                np.linalg.norm(pos - self._prev_position)
            )
        self._prev_position = pos.copy()
        self.steps_since_new_node += 1

        # Check revisit
        nearest_id, nearest_dist = self.find_nearest_node(pos)
        if nearest_id is not None and nearest_dist < self.cfg.revisit_radius:
            self.visit_counts[nearest_id] += 1
            self.graph.nodes[nearest_id]["visit_count"] = self.visit_counts[
                nearest_id
            ]
            if (
                self.last_node_id is not None
                and self.last_node_id != nearest_id
                and not self.graph.has_edge(self.last_node_id, nearest_id)
            ):
                w = max(self.distance_since_last_node, nearest_dist)
                self.graph.add_edge(self.last_node_id, nearest_id, weight=w)
            self.last_node_id = nearest_id
            self.distance_since_last_node = 0.0
            return None

        # Place new node?
        if nearest_id is None or nearest_dist >= self.cfg.waypoint_spacing:
            nid = self._add_node(pos)
            if self.last_node_id is not None:
                w = max(self.distance_since_last_node, 1e-6)
                self.graph.add_edge(self.last_node_id, nid, weight=w)
            # Mark as frontier if lots of open space
            if depth_analysis["open_fraction"] > 0.4:
                self.backtrack_stack.append(nid)
            self.last_node_id = nid
            self.distance_since_last_node = 0.0
            self.steps_since_new_node = 0
            return nid

        return None

    # -- coverage ---------------------------------------------------------

    def estimate_coverage(self, pathfinder, n_samples: int = 200) -> float:
        if not self.node_positions:
            return 0.0
        pts = np.array(list(self.node_positions.values()))
        radius = self.cfg.waypoint_spacing * 2
        covered = 0
        for _ in range(n_samples):
            sample = pathfinder.get_random_navigable_point()
            dists = np.linalg.norm(pts - np.array(sample), axis=1)
            if dists.min() < radius:
                covered += 1
        return covered / n_samples


# ---------------------------------------------------------------------------
# Step 5 – Action selection
# ---------------------------------------------------------------------------


def _agent_forward(agent_state: habitat_sim.AgentState) -> np.ndarray:
    """Return the agent's forward direction as a unit vector in xz-plane."""
    fwd = utils.quat_to_magnum(agent_state.rotation).transform_vector(
        mn.Vector3(0, 0, -1.0)
    )
    return np.array([fwd[0], fwd[1], fwd[2]], dtype=float)


def compute_turn_or_forward_toward(
    target_pos: np.ndarray,
    agent_state: habitat_sim.AgentState,
    cfg: ExplorationConfig,
) -> str:
    """Return the discrete action that steers toward *target_pos*."""
    agent_pos = np.array(agent_state.position, dtype=float)
    to_target = target_pos - agent_pos
    to_target[1] = 0.0  # project onto xz
    dist = np.linalg.norm(to_target)
    if dist < 1e-6:
        return "move_forward"
    to_target /= dist

    fwd = _agent_forward(agent_state)
    fwd_xz = np.array([fwd[0], 0.0, fwd[2]])
    n = np.linalg.norm(fwd_xz)
    if n < 1e-6:
        return "move_forward"
    fwd_xz /= n

    # Signed angle (positive = need to turn left in Habitat convention)
    cross_y = fwd_xz[0] * to_target[2] - fwd_xz[2] * to_target[0]
    dot = np.clip(fwd_xz[0] * to_target[0] + fwd_xz[2] * to_target[2], -1, 1)
    angle_deg = math.degrees(math.acos(dot))

    if angle_deg < cfg.turn_amount / 2:
        return "move_forward"
    if cross_y > 0:
        return "turn_left" if angle_deg > 15 else "turn_left_small"
    return "turn_right" if angle_deg > 15 else "turn_right_small"


def choose_action(
    depth_analysis: Dict[str, Any],
    agent_state: habitat_sim.AgentState,
    memory: ExplorationMemory,
    mode: ExplorationState,
    sim: habitat_sim.Simulator,
    cfg: ExplorationConfig,
    rng: np.random.Generator,
    random_walk_remaining: int,
) -> Tuple[str, ExplorationState, int]:
    """Return ``(action, new_mode, random_walk_remaining)``."""

    # -- RANDOM WALK ------------------------------------------------------
    if mode == ExplorationState.RANDOM_WALK:
        action = rng.choice(
            ["move_forward", "move_forward", "turn_left", "turn_right"]
        )
        random_walk_remaining -= 1
        if random_walk_remaining <= 0:
            mode = ExplorationState.EXPLORE
        return action, mode, random_walk_remaining

    # -- BACKTRACK --------------------------------------------------------
    if mode == ExplorationState.BACKTRACK:
        # Pop exhausted targets
        agent_pos = np.array(agent_state.position, dtype=float)
        while memory.backtrack_stack:
            tid = memory.backtrack_stack[-1]
            target = memory.node_positions[tid]
            if np.linalg.norm(agent_pos - target) < cfg.revisit_radius:
                memory.backtrack_stack.pop()
            else:
                break
        if not memory.backtrack_stack:
            return (
                "move_forward",
                ExplorationState.RANDOM_WALK,
                rng.integers(30, 51),
            )

        target = memory.node_positions[memory.backtrack_stack[-1]]

        # Use pathfinder for next waypoint
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = target
        if sim.pathfinder.find_path(path) and len(path.points) >= 2:
            action = compute_turn_or_forward_toward(
                np.array(path.points[1]), agent_state, cfg
            )
        else:
            memory.backtrack_stack.pop()
            action = rng.choice(["turn_left", "turn_right"])

        return action, mode, random_walk_remaining

    # -- EXPLORE (default) ------------------------------------------------
    bd = depth_analysis["best_direction"]

    if depth_analysis["obstacle_ahead"]:
        if bd < -0.2:
            action = "turn_left"
        elif bd > 0.2:
            action = "turn_right"
        else:
            # Dead end – backtrack
            if memory.backtrack_stack:
                return (
                    "turn_left",
                    ExplorationState.BACKTRACK,
                    random_walk_remaining,
                )
            return (
                "turn_left",
                ExplorationState.RANDOM_WALK,
                rng.integers(30, 51),
            )
    else:
        if abs(bd) < 0.15:
            action = "move_forward"
        elif bd < 0:
            action = "turn_left_small"
        else:
            action = "turn_right_small"

    # Stall detection
    if memory.steps_since_new_node > cfg.stall_limit // 2:
        mode = ExplorationState.RANDOM_WALK
        random_walk_remaining = rng.integers(30, 51)

    return action, mode, random_walk_remaining


# ---------------------------------------------------------------------------
# Step 6 – Main exploration loop
# ---------------------------------------------------------------------------


def explore_scene(cfg: ExplorationConfig) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Run the exploration and return ``(graph, metadata)``."""
    sim, agent = create_simulator(cfg)
    try:
        return _explore_loop(sim, agent, cfg)
    finally:
        sim.close()


def _explore_loop(
    sim: habitat_sim.Simulator,
    agent: habitat_sim.Agent,
    cfg: ExplorationConfig,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    memory = ExplorationMemory(cfg)
    memory._pathfinder = sim.pathfinder

    # Initial node
    start_pos = np.array(agent.get_state().position, dtype=float)
    memory._add_node(start_pos)
    memory.last_node_id = 0
    memory._prev_position = start_pos.copy()

    mode = ExplorationState.EXPLORE
    rw_remaining = 0
    termination_reason = "max_steps"

    for step in range(cfg.max_steps):
        obs = sim.get_sensor_observations()
        depth = obs["depth_sensor"]

        # Guard degenerate depth
        if depth is None or depth.size == 0 or not np.any(np.isfinite(depth)):
            sim.step(rng.choice(["turn_left", "turn_right"]))
            continue

        analysis = analyze_depth_image(depth, cfg)
        agent_state = agent.get_state()
        pos = np.array(agent_state.position, dtype=float)

        # Snap if NaN
        if np.any(np.isnan(pos)):
            snapped = sim.pathfinder.snap_point(pos)
            if not np.any(np.isnan(snapped)):
                state = habitat_sim.AgentState()
                state.position = snapped
                state.rotation = agent_state.rotation
                agent.set_state(state)
                pos = np.array(snapped, dtype=float)
            else:
                sim.step("turn_left")
                continue

        memory.update(pos, analysis)

        # Termination checks
        if memory.steps_since_new_node >= cfg.stall_limit:
            termination_reason = "stall_limit"
            break
        if step > 0 and step % 100 == 0:
            cov = memory.estimate_coverage(sim.pathfinder)
            if cov >= 0.85:
                termination_reason = f"coverage_{cov:.2f}"
                break

        action, mode, rw_remaining = choose_action(
            analysis, agent_state, memory, mode, sim, cfg, rng, rw_remaining
        )
        sim.step(action)

    n = memory.graph.number_of_nodes()
    e = memory.graph.number_of_edges()
    if n < 2:
        logger.warning(
            "Exploration produced only %d node(s). "
            "Consider increasing --max-steps or decreasing --waypoint-spacing.",
            n,
        )

    metadata = {
        "scene_name": cfg.scene_name,
        "num_nodes": n,
        "num_edges": e,
        "total_steps": step + 1 if 'step' in dir() else 0,
        "termination_reason": termination_reason,
        "config": asdict(cfg),
    }
    logger.info(
        "Exploration complete: %d nodes, %d edges, reason=%s",
        n,
        e,
        termination_reason,
    )
    return memory.graph, metadata


# ---------------------------------------------------------------------------
# Step 7 – Serialization
# ---------------------------------------------------------------------------


def _output_dir(cfg: ExplorationConfig) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = Path(cfg.output_dir) / f"{cfg.scene_name}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_graph(
    G: nx.Graph, metadata: Dict[str, Any], cfg: ExplorationConfig
) -> Dict[str, Path]:
    """Save the graph as pickle, GraphML, and metadata JSON."""
    out = _output_dir(cfg)
    paths: Dict[str, Path] = {}

    # Pickle
    pkl_path = out / "graph.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    paths["pickle"] = pkl_path

    # GraphML – convert numpy arrays to scalar attributes
    G_ml = G.copy()
    for nid in G_ml.nodes:
        p = G_ml.nodes[nid].get("pos")
        if p is not None:
            G_ml.nodes[nid]["x"] = float(p[0])
            G_ml.nodes[nid]["y"] = float(p[1])
            G_ml.nodes[nid]["z"] = float(p[2])
            del G_ml.nodes[nid]["pos"]
        p2 = G_ml.nodes[nid].get("pos_2d")
        if p2 is not None:
            G_ml.nodes[nid]["pos_2d_x"] = float(p2[0])
            G_ml.nodes[nid]["pos_2d_y"] = float(p2[1])
            del G_ml.nodes[nid]["pos_2d"]
    gml_path = out / "graph.graphml"
    nx.write_graphml(G_ml, str(gml_path))
    paths["graphml"] = gml_path

    # Metadata JSON
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    paths["metadata"] = meta_path

    return paths


# ---------------------------------------------------------------------------
# Step 8 – Visualization
# ---------------------------------------------------------------------------


def convert_points_to_topdown(
    pathfinder, points: List[np.ndarray], meters_per_pixel: float
) -> List[np.ndarray]:
    """Convert 3-D world points to 2-D pixel coordinates on a top-down map."""
    bounds = pathfinder.get_bounds()
    out = []
    for pt in points:
        px = (pt[0] - bounds[0][0]) / meters_per_pixel
        py = (pt[2] - bounds[0][2]) / meters_per_pixel
        out.append(np.array([px, py]))
    return out


def visualize_topdown_overlay(
    G: nx.Graph,
    sim: habitat_sim.Simulator,
    cfg: ExplorationConfig,
    save_path: Optional[Path] = None,
) -> None:
    """Draw graph nodes/edges on top of the scene's top-down navigability map."""
    meters_per_pixel = 0.05
    height = sim.pathfinder.get_bounds()[0][1]
    tdmap = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

    points_3d = [G.nodes[n]["pos"] for n in sorted(G.nodes)]
    pts_2d = convert_points_to_topdown(sim.pathfinder, points_3d, meters_per_pixel)
    visits = [G.nodes[n].get("visit_count", 1) for n in sorted(G.nodes)]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(tdmap, cmap="gray", origin="upper")

    # Edges
    id_to_idx = {n: i for i, n in enumerate(sorted(G.nodes))}
    for u, v in G.edges:
        p1, p2 = pts_2d[id_to_idx[u]], pts_2d[id_to_idx[v]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "c-", linewidth=0.8, alpha=0.7)

    # Nodes
    xs = [p[0] for p in pts_2d]
    ys = [p[1] for p in pts_2d]
    sc = ax.scatter(xs, ys, c=visits, cmap="hot", s=20, zorder=5, edgecolors="k",
                    linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="visit count", shrink=0.6)
    ax.set_title(f"Top-down overlay – {cfg.scene_name}")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_networkx_topology(
    G: nx.Graph, save_path: Optional[Path] = None
) -> None:
    """Spatial and spring-layout views of graph topology."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    pos_spatial = {n: G.nodes[n]["pos_2d"] for n in G}
    visits = np.array([G.nodes[n].get("visit_count", 1) for n in G])
    node_sizes = 30 + 20 * visits

    weights = np.array(
        [G.edges[e].get("weight", 1.0) for e in G.edges], dtype=float
    )
    max_w = weights.max() if weights.size and weights.max() > 0 else 1.0
    edge_widths = 0.5 + 2.0 * (1.0 - weights / max_w)

    nx.draw_networkx(
        G,
        pos=pos_spatial,
        ax=ax1,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=False,
        node_color=visits,
        cmap=plt.cm.viridis,
        edge_color="gray",
    )
    ax1.set_title("Spatial layout (x, z)")
    ax1.set_aspect("equal")

    pos_spring = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G,
        pos=pos_spring,
        ax=ax2,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=False,
        node_color=visits,
        cmap=plt.cm.viridis,
        edge_color="gray",
    )
    ax2.set_title("Spring layout (topology)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_3d(G: nx.Graph, save_path: Optional[Path] = None) -> None:
    """3-D scatter + edges with Habitat y-up → matplotlib z-up swap."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    positions = {n: G.nodes[n]["pos"] for n in G}
    elevations = [float(positions[n][1]) for n in G]

    xs = [float(positions[n][0]) for n in G]
    ys = [float(positions[n][2]) for n in G]  # Habitat z → mpl y
    zs = [float(positions[n][1]) for n in G]  # Habitat y → mpl z

    ax.scatter(xs, ys, zs, c=elevations, cmap="coolwarm", s=20, depthshade=True)

    for u, v in G.edges:
        pu, pv = positions[u], positions[v]
        ax.plot(
            [pu[0], pv[0]],
            [pu[2], pv[2]],
            [pu[1], pv[1]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
        )

    ax.view_init(elev=75, azim=-90)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Elevation (Y)")
    ax.set_title("3D navigation graph")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_all(
    G: nx.Graph,
    sim: habitat_sim.Simulator,
    cfg: ExplorationConfig,
    out_dir: Path,
) -> None:
    """Generate and save all three visualizations."""
    visualize_topdown_overlay(G, sim, cfg, save_path=out_dir / "topdown.png")
    visualize_networkx_topology(G, save_path=out_dir / "topology.png")
    visualize_3d(G, save_path=out_dir / "3d.png")
    logger.info("Visualizations saved to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--scene-name",
    type=click.Choice(list(SCENE_REGISTRY.keys())),
    default="apartment_1",
    show_default=True,
    help="Scene to explore.",
)
@click.option("--data-path", default="data/habitat/", show_default=True)
@click.option("--max-steps", type=int, default=2000, show_default=True)
@click.option("--waypoint-spacing", type=float, default=0.5, show_default=True)
@click.option("--revisit-radius", type=float, default=0.3, show_default=True)
@click.option("--stall-limit", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--forward-amount", type=float, default=0.25, show_default=True)
@click.option("--turn-amount", type=float, default=10.0, show_default=True)
@click.option("--output-dir", default="logs/nav_graphs/", show_default=True)
@click.option("--no-viz", is_flag=True, default=False, help="Skip visualization.")
def main(
    scene_name: str,
    data_path: str,
    max_steps: int,
    waypoint_spacing: float,
    revisit_radius: float,
    stall_limit: int,
    seed: int,
    forward_amount: float,
    turn_amount: float,
    output_dir: str,
    no_viz: bool,
) -> None:
    """Generate a navigation graph by exploring a Habitat-sim scene."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = ExplorationConfig(
        scene_name=scene_name,
        data_path=data_path,
        max_steps=max_steps,
        waypoint_spacing=waypoint_spacing,
        revisit_radius=revisit_radius,
        stall_limit=stall_limit,
        seed=seed,
        forward_amount=forward_amount,
        turn_amount=turn_amount,
        output_dir=output_dir,
    )

    click.echo(f"Exploring scene '{cfg.scene_name}' for up to {cfg.max_steps} steps …")
    G, metadata = explore_scene(cfg)

    out = _output_dir(cfg)
    saved = save_graph(G, metadata, cfg)
    click.echo(
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    for fmt, path in saved.items():
        click.echo(f"  {fmt}: {path}")

    if not no_viz:
        click.echo("Generating visualizations …")
        sim, _ = create_simulator(cfg)
        try:
            visualize_all(G, sim, cfg, out)
        finally:
            sim.close()
        click.echo(f"Visualizations saved to {out}")


if __name__ == "__main__":
    main()
