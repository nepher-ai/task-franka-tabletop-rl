# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded episode in ``task-franka-tabletop``'s own evaluation environment.

A recorded action is an absolute joint target, so the original robot, physics, and scene must follow
the recorded joint path and finish the same task. This is the check that the dataset is valid
training data for brains that are scored there. Run it on an episode recorded with ``--jitter 0``:
the original environment lays the published scene out without jitter.

Example:
    python scripts/replay_check.py --dataset datasets/tabletop_rl --episode 0
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded episode in the original tabletop environment.")
parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset directory.")
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
parser.add_argument("--max_joint_error", type=float, default=0.15, help="Largest allowed |joint error| [rad].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# The original environment always mounts its cameras.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from franka_tabletop.core.eval_env import TabletopEvalEnv  # noqa: E402
from franka_tabletop.core.observations import gripper_pos, joint_pos  # noqa: E402
from franka_tabletop.env_cfg import _apply_benchmark_assets, build_cfg  # noqa: E402
from franka_tabletop_rl.benchmark.bundle import load_benchmark, resolve_bundle  # noqa: E402
from franka_tabletop_rl.benchmark.legacy_layout import legacy_bundle_view  # noqa: E402
from franka_tabletop_rl.data.lerobot_writer import CHUNK_SIZE, DATA_PATH  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401


def main() -> None:
    with open(os.path.join(args_cli.dataset, "meta", "episodes.jsonl"), encoding="utf-8") as stream:
        meta = [json.loads(line) for line in stream]
    info = next(entry for entry in meta if entry["episode_index"] == args_cli.episode)
    task_id = info["task_id"]
    chunk = args_cli.episode // CHUNK_SIZE
    table = pq.read_table(
        os.path.join(args_cli.dataset, DATA_PATH.format(episode_chunk=chunk, episode_index=args_cli.episode))
    ).to_pydict()
    actions = np.asarray(table["action"], dtype=np.float32)
    states = np.asarray(table["observation.state"], dtype=np.float32)

    bundle = resolve_bundle()
    benchmark = load_benchmark(bundle)
    scene = str(benchmark.record(task_id)["layout"])
    view = legacy_bundle_view(bundle, tempfile.mkdtemp(prefix="tabletop-replay-"))
    cfg = build_cfg(task_id=task_id, scene=scene, num_envs=1, observation={"cameras": []})
    cfg.bundle_path = str(view)
    _apply_benchmark_assets(cfg)
    env = TabletopEvalEnv(cfg)
    env.reset()

    worst = 0.0
    for step, action in enumerate(actions):
        recorded = states[step]
        replayed = torch.cat([joint_pos(env), gripper_pos(env)], dim=-1)[0].cpu().numpy()
        worst = max(worst, float(np.abs(recorded[:7] - replayed[:7]).max()))
        env.step(torch.as_tensor(action[None], device=env.device))
    completed = bool(env.task_completed()[0])
    print(
        f"[RESULT] replay of episode {args_cli.episode} ({task_id}): completed={completed}, max error {worst:.3f} rad"
    )
    env.close()
    if not completed or worst > args_cli.max_joint_error:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
