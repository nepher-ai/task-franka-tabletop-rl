# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record successful planner runs as a LeRobot v2.1 dataset.

Every selected task runs in one stage. A step stores the brain observation and the absolute joint
target the tracker applied.
"""

# Launch Isaac Sim first. Modules that import pxr must load after the app starts.

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record planner demonstrations as a LeRobot dataset.")
parser.add_argument("--task", type=str, default=None, help="One benchmark task id. Every task when omitted.")
parser.add_argument("--output", type=str, required=True, help="Dataset directory. An existing dataset is extended.")
parser.add_argument("--episodes", type=int, default=8, help="Successful episodes to record per task.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Parallel envs. A multiple of the task count. One copy of each task by default.",
)
parser.add_argument("--jitter", type=float, default=0.00, help="Per-episode shift [m] of each object.")
parser.add_argument("--seed", type=int, default=1, help="Seed of the first layout.")
parser.add_argument(
    "--cameras",
    nargs="*",
    default=None,
    metavar="NAME",
    help=(
        "Cameras to record: cam_exterior_left, cam_exterior_right, cam_wrist, cam_front. "
        "Pass the flag with no names to record every camera. Omit the flag to record none."
    ),
)
parser.add_argument("--image_hw", type=int, nargs=2, default=(180, 320), help="Camera frame size [pixels]: H W.")
parser.add_argument("--max_steps", type=int, default=4800, help="Give up after this many control steps.")
parser.add_argument("--keep_failed", action="store_true", default=False, help="Also save episodes that fail.")
parser.add_argument("--checkpoint", type=str, default=None, help="Tracker checkpoint file.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.cameras is not None:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that need the simulation app follow.

import franka_tabletop_rl.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from franka_tabletop_rl.data.lerobot_writer import LeRobotWriter  # noqa: E402
from franka_tabletop_rl.paths import sync_best_policy  # noqa: E402
from franka_tabletop_rl.benchmark.bundle import ALL_TASKS  # noqa: E402
from franka_tabletop_rl.scene.cameras import OBSERVATION_CAMERAS, selected_cameras  # noqa: E402
from franka_tabletop_rl.scene.observations import camera_rgb, eef_pose, gripper_pos, joint_pos, resize_rgb  # noqa: E402
from franka_tabletop_rl.scene.robot import GRIPPER_CLOSED_RAD  # noqa: E402
from franka_tabletop_rl.tasks.hl_policy.hl_env_cfg import HLEnvCfg_PLAY  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

GYM_ID = "Nepher-FrankaTabletopRL-HL-Play-v0"


def _host(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def main() -> None:
    installed_version = check_rsl_rl_version()
    agent_cfg = load_cfg_from_registry(GYM_ID, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    checkpoint = sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=args_cli.checkpoint,
    )

    # A bare ``--cameras`` is an empty list and means every camera. Omitting the flag is None.
    cameras = OBSERVATION_CAMERAS if args_cli.cameras == [] else selected_cameras(args_cli.cameras)
    cfg = HLEnvCfg_PLAY(
        task_id=args_cli.task or ALL_TASKS,
        pose_jitter_m=args_cli.jitter,
        cameras=cameras,
        layout_seed=args_cli.seed,
    )
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(GYM_ID, cfg=cfg)
    unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)

    tracker = unwrapped.command_manager.get_term("ee_pose").tracker
    arm_action = unwrapped.action_manager.get_term("arm_action")
    size = tuple(args_cli.image_hw)
    fps = round(1.0 / unwrapped.step_dt)
    writer = LeRobotWriter(args_cli.output, fps=fps, camera_hw={name: size for name in cameras})

    count = unwrapped.num_envs
    buffers = [{"state": [], "action": [], "eef": [], **{name: [] for name in cameras}} for _ in range(count)]
    saved = {task_id: 0 for task_id in tracker.task_ids}
    tried, success_count = 0, 0
    obs = env.get_observations()
    for step in range(args_cli.max_steps):
        state = torch.cat([joint_pos(unwrapped), gripper_pos(unwrapped)], dim=-1)
        pose = eef_pose(unwrapped)
        images = {name: resize_rgb(camera_rgb(unwrapped, name), size) for name in cameras}
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        command = arm_action.joint_command
        action = torch.cat([command[:, :7], (command[:, 7:8] > 0.5 * GRIPPER_CLOSED_RAD).float()], dim=-1)
        state_h, pose_h, action_h = _host(state), _host(pose), _host(action)
        images_h = {name: _host(frames) for name, frames in images.items()}
        done_h = _host(dones).astype(bool)
        for index in range(count):
            buffer = buffers[index]
            buffer["state"].append(state_h[index])
            buffer["eef"].append(pose_h[index])
            buffer["action"].append(action_h[index])
            for name in cameras:
                buffer[name].append(images_h[name][index])
            if not done_h[index]:
                continue
            tried += 1
            task_id = tracker.last_task_ids[index]
            instruction = next(layout.instruction for layout in tracker.task_layouts if layout.task_id == task_id)
            success = bool(tracker.last_success[index])
            success_count += int(success)
            if saved[task_id] < args_cli.episodes and (success or args_cli.keep_failed):
                writer.add_episode(
                    np.stack(buffer["state"]),
                    np.stack(buffer["action"]),
                    np.stack(buffer["eef"]),
                    task=instruction,
                    frames={name: np.stack(buffer[name]) for name in cameras},
                    extra={
                        "success": success,
                        "task_id": task_id,
                        "progress": float(tracker.last_progress[index]),
                    },
                )
                saved[task_id] += 1
                steps = len(buffer["action"])
                print(f"[INFO] {task_id}: episode {saved[task_id]} saved ({steps} steps, {success=})", flush=True)
            for key in buffer:
                buffer[key] = []
        if all(episodes >= args_cli.episodes for episodes in saved.values()):
            break
    for task_id, episodes in saved.items():
        print(f"[RESULT] {task_id}: {episodes} episodes saved")
    print(f"[RESULT] {success_count}/{tried} episodes succeeded, {sum(saved.values())} saved to {args_cli.output}")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
