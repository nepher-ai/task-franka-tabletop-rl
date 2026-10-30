# Franka tabletop pick-and-place in NVIDIA Isaac Lab

**classical planner · learned end-effector tracker · damped least-squares IK · 8-D joint targets**

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![rsl_rl](https://img.shields.io/badge/rsl__rl-PPO-orange.svg)](https://github.com/leggedrobotics/rsl_rl)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

![Franka tabletop pick-and-place scenes in Isaac Sim](images/image.png)

**task-franka-tabletop-rl** is a [Nepher Robotics](https://github.com/nepher-ai) extension for [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab) / [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html). It trains a **Franka** arm with a **Robotiq 2F-85** gripper to pick and place objects on a tabletop benchmark. A classical planner chooses each grasp and place pose. An **RSL-RL PPO** policy tracks that tool pose. **Damped least-squares IK** turns the tracker action into **8-D absolute joint targets**.

Use this repository when you need Isaac Lab Gym environments for Franka **tabletop pick-and-place** (`Nepher-FrankaTabletopRL-LL-*`, `Nepher-FrankaTabletopRL-HL-*`), a LeRobot v2.1 dataset of successful episodes, or the EnvHub benchmarks `franka-tabletop-rl-v0`, `franka-tabletop-rl-v1`, `franka-tabletop-rl-v2`, and `franka-tabletop-rl-v3`.

Agents: start at [`llms.txt`](llms.txt) (index) or [`llms-full.txt`](llms-full.txt) (single-file context).

| | |
|---|---|
| **Robot** | Franka with a Robotiq 2F-85 gripper |
| **Simulator** | NVIDIA Isaac Sim 6.1.0 + Isaac Lab 3.0 |
| **Algorithm** | rsl_rl PPO end-effector tracker |
| **Stack** | Classical planner → PPO tracker → damped least-squares IK → 8-D joint targets |
| **Rates** | Physics 120 Hz. Commands every 8 steps (15 Hz) |
| **Tracker observation** | 43 values: arm position and velocity, gripper, tool pose, pose error, pose target, grip target, last action |
| **Tracker action** | 7-D tool-pose delta plus gripper, integrated to absolute joint targets |
| **Recorded action** | 8 values: 7 absolute joint targets [rad], gripper 0 (open) or 1 (closed) |
| **Success** | The task predicate holds on the current state for 0.5 s |
| **Eval** | EnvHub `franka-tabletop-rl-v0` (default), plus `v1`, `v2`, and `v3` |
| **License** | BSD 3-Clause |

## Who this is for

- Researchers training a **Franka end-effector tracker** in Isaac Lab, then running it under a classical pick-and-place planner.
- Teams collecting a **LeRobot v2.1** dataset of successful tabletop episodes, with joint targets and the four scene cameras.
- Benchmarks on the EnvHub bundles **`franka-tabletop-rl-v0`** through **`franka-tabletop-rl-v3`**. The high-level env puts every selected task in one stage.

The package collects training data for a downstream policy. It does not open a socket and it does not ship a brain image. The tournament scene those policies are scored on is `task-franka-tabletop`.

## How it runs

```
classical planner (hl_policy) -> learned end-effector tracker (ll_policy) -> damped-least-squares IK -> 8-D joint targets
```

Train the tracker first, then play the planner on top of that checkpoint.

- **High level.** `ClassicalTabletopPlanner` picks each object a task asks to move. It searches a top-down grasp on the object's mesh, carries the object to the pose that satisfies that goal, lets go, and retries a failed grasp. A task that must be left upright is left alone and scores 0. The planner reads latched benchmark progress and stops when the task is done.
- **Low level.** The PPO policy tracks the planner's tool pose and gripper command. Its 7-D action is a tool-pose delta plus the gripper. `IkJointCommandAction` turns that into absolute joint targets, the same 8-D action space a downstream policy writes.
- **Scene.** The robot, physics, sensors, and success predicates match `task-franka-tabletop` at commit `4d64971`. They live in `franka_tabletop_rl/scene/` and `franka_tabletop_rl/benchmark/`. `tests/test_parity.py` checks them, so a recorded trajectory is valid training data for that benchmark.

### Episode success

The success expression has to hold on the current state, with no memory of earlier grasps or placements, and it has to keep holding for 0.5 s. An object that must be moved has to be free of the fingers (a stack is exempt), and a pickup has to still be in them. `in_order` keeps only the first-placement step of each object. Contact-force checks are not ported, so `stacked` also needs the lowest corner within 1 cm of the support top. `front` is toward the robot (-x) and `behind` is away from it (+x). A lost object ends the episode as a failure and never counts as success. The checks are in `franka_tabletop_rl/benchmark/success.py`.

## Gym environments

| Id | Use |
| --- | --- |
| `Nepher-FrankaTabletopRL-LL-v0` | Train the tracker (4096 envs, observation noise) |
| `Nepher-FrankaTabletopRL-LL-Play-v0` | Play the tracker |
| `Nepher-FrankaTabletopRL-HL-Play-v0` | Planner plus tracker on every selected task in one stage. Loads the latest tracker checkpoint |
| `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` | The same layout, for eval-nav. Loads `best_policy/best_policy.pt` |

`./isaaclab.sh train` and `play` discover these through the `isaaclab.tasks` entry point.

Both high-level tasks load the EnvHub benchmark named by `env_id`. The default is `franka-tabletop-rl-v0` (`nepher download franka-tabletop-rl-v0`), 16 tasks. `franka-tabletop-rl-v1` is the full task set those smaller bundles are drawn from. `franka-tabletop-rl-v2` is 16 tasks taken from v1, one for each distinct set of success ops, including `one_bottle_in_square_pail`. `franka-tabletop-rl-v3` covers those same op sets and uses the next v1 task after v2 wherever v1 has another one. `one_bottle_in_square_pail` stays in v3.

**One stage, all tasks.** The high-level env builds one stage for the selected tasks (`env.task_id=all`, or one task id). Each benchmark has one background, named by its stage. `scene.num_envs` defaults to one env per task. `--num_envs` is that count or a whole multiple of it (`16`, `32`, `64`, ... for v0, v2, and v3). Env `i` runs task `i % T` and resets on its own, and every task is in the stage together. Unused object positions rest above the cameras. Because every env owns its meshes, the cloner builds one world per env (`replicate_physics=False`).

## Install

Isaac Sim has to run with its bundled Python, so every command goes through `./isaaclab.sh` from the Isaac Lab root. Inside that tree this package is `source/task-franka-tabletop-rl/source/franka_tabletop_rl`.

```bash
./isaaclab.sh -p -m pip install -e source/task-franka-tabletop-rl/source/franka_tabletop_rl
./isaaclab.sh -p -m pip install -e "source/task-franka-tabletop-rl/source/franka_tabletop_rl[record]"
```

The second line adds the LeRobot writer (`pyarrow`, `av`). TensorBoard logs are under this project, at `logs/rsl_rl/franka_tabletop_ll_ee_tracking`, not the Isaac Lab `logs` directory.

## Train and play

Isaac Lab 3 does not have a `--headless` flag. `--viz none` runs without a window. Playback runs until Ctrl+C. `--num_envs` for a high-level task defaults to one env per task, and a larger value has to be a multiple of that count. One task is `env.task_id=<task_id>`.

```bash
./isaaclab.sh train --task Nepher-FrankaTabletopRL-LL-v0 --num_envs 4096 --max_iterations 1500 --viz none
./isaaclab.sh play --task Nepher-FrankaTabletopRL-LL-Play-v0 --num_envs 64 --viz none
./isaaclab.sh play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --viz none
./isaaclab.sh play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 4 env.task_id=throw_away_apple --viz none
```

`Nepher-FrankaTabletopRL-LL-Play-v0` and `Nepher-FrankaTabletopRL-HL-Play-v0` load the latest `model_*.pt` under `logs/rsl_rl/franka_tabletop_ll_ee_tracking` unless you pass `--checkpoint`. Copy the checkpoint eval-nav should load to `best_policy/best_policy.pt`. `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` loads that file. The checkpoint is not committed.

### Highlight videos

`env.highlight=on` on `Nepher-FrankaTabletopRL-HL-Play-v0` or `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` adds video recording and changes nothing else: the tasks, `--num_envs`, and layouts stay those of the same command. It writes `best_policy/videos/highlight/<camera>.mp4` for `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, and `cam_front`. Each is 1280x720, a 4x4 grid of 320x180 cells, and lasts one episode. The grid shows up to 16 tasks of the stage in benchmark order. A stage with more than 16 tasks shows 16 sampled with the seed. A stage with fewer leaves the remaining cells black. `env.highlight_tasks=0-15` films that inclusive range, and `0,5,12` films those indices, in that order instead of the sample. Only those envs are rendered, at the cell size. The process exits after the four clips are written.

```bash
./isaaclab.sh play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --viz none env.highlight=on
./isaaclab.sh play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --viz none env.highlight=on "env.highlight_tasks=0-15"
```

### Film shots

`scripts/record_film.py` records the shots of the Franka video offscreen, with no UI or gizmos, at 1920x1080 and 60 fps (the close-up is 3840x2160). It runs the tracker checkpoint from `best_policy/best_policy.pt` and writes `film/<shot>_<pass>.mp4` and a `<shot>_<pass>.json` next to it. The JSON holds the cut points, the planner events, and whether the replay matched the scout. Each take has 2 s before and 3 s after the part to use. `--fps` must divide the 120 Hz physics rate, and `--resolution 1920x1080` overrides the size. The 4x speed-up of the long run and any slow motion are applied when editing. `film/` is not committed.

```bash
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/record_film.py list
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/record_film.py scout failure
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/record_film.py preview failure
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/record_film.py shoot failure
```

A take is made in two runs, except the parallel grid, which needs no scout. `scout` runs the stage without rendering and logs every episode to `film/scout/`. It stops at the first episode that can supply the shot, or when every env has finished `--episodes_per_env` episodes, and `--keep_scouting` goes on to that limit. Shots on the same stage share one scout log, so scout with `--keep_scouting` once to serve several of them. `shoot` runs the same stage again, renders the chosen episode, and checks that it played out as scouted. Scout and shoot need the same checkpoint, or `shoot` stops unless you pass `--force`. `--rank 1` takes the second-best episode, and `--env 3 --episode 1` takes that episode of that env. `preview` is `shoot` at 960x540 and 15 fps with PNG stills at the camera keys and cut points, written to `film/preview`.

| # | Shot | Command | Take |
| --- | --- | --- | --- |
| 1 | Hero grasp | `scout hero`, then `shoot hero` | Low table-height camera, side-on, on `stack_yellow_on_red`, the carry comes toward the lens. 3 s before the first close, 4.5 s after |
| 2 | Busy table | `scout busy_table`, then `shoot busy_table` | Slow high-and-wide push-in on `grab_a_bagel`, 5 s from the reset |
| 3 | Point cloud | `shoot busy_table --pass points` | Same camera and take as shot 2, for the cross-fade. The planner's per-object surface points at the live object poses, with markers for the tool, the grasp and release poses, and the place goal |
| 4 | Long run | `scout long_run`, then `shoot long_run` | One 15-40 s episode of a pack or stack task, fixed camera. Play it at 4x. The tracker needs 15 to 17 s at best on these tasks |
| 5 | Failure and success | `scout failure`, `shoot failure`, `shoot success` | Same task and camera. The failure is a natural one found by the scout, never staged. 5 s each |
| 6 | Gripper close-up | `scout closeup`, then `shoot closeup` | 3840x2160 camera that follows the tool, 2.5 s. `--fps 120 --resolution 1920x1080` gives slow motion |
| 7 | Parallel grid | `shoot parallel` | All envs in one overhead view from the first step, 5 s |

Shots 2 and 3 must use the same `--rank` or `--env` and `--episode`. Every mode except `list` needs a trained checkpoint. `scripts/film_framing.py <shot>` draws wireframe stills of a shot's camera to `film/framing` without Isaac Sim. Those are a schematic of the layout and not the render, so use `preview` to check the real framing.

### LeRobot dataset

`scripts/record_lerobot.py` runs the planner and the tracker, and keeps the successful episodes as a LeRobot v2.1 dataset (`--cameras` records every camera as h264 video, and `--cameras cam_wrist cam_front` records those). It records all tasks at once, or one with `--task <task_id>`. `--episodes` is the number of successful episodes kept per task. `--num_envs` defaults to one env per task, and a larger value must be a multiple of the task count. It also copies the tracker it used to `best_policy/best_policy.pt`.

```bash
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/record_lerobot.py --episodes 8 --output datasets/tabletop_rl
./isaaclab.sh -p source/task-franka-tabletop-rl/scripts/replay_check.py --dataset datasets/tabletop_rl --episode 0
```

| Feature | Shape | Content |
| --- | --- | --- |
| `observation.state` | 8 | 7 joint positions [rad], gripper closure 0..1 |
| `observation.eef_pose` | 7 | tool position [m] and wxyz quaternion |
| `action` | 8 | 7 absolute joint targets [rad], gripper 0 (open) or 1 (closed) |
| `observation.images.<camera>` | H, W, 3 | `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, `cam_front` |

`meta/modality.json` declares the GR00T embodiment `franka_2f85_jointpos`. `scripts/replay_check.py` replays an episode recorded with `--jitter 0` in `task-franka-tabletop`'s own `TabletopEvalEnv`.

## FAQ

**What robot and simulator does this target?**
A Franka arm with a Robotiq 2F-85 gripper, in NVIDIA Isaac Sim 6.1.0 with Isaac Lab 3.0. Physics steps at 120 Hz. The policy command lands every 8 steps, at 15 Hz.

**What does the tracker observe and output?**
The observation is 43 values: 7 arm positions, 7 arm velocities, gripper, 7-D tool pose, 6-D pose error, 7-D pose target, grip target, and the last 7-D action. The action is a 7-D tool-pose delta plus the gripper. `IkJointCommandAction` integrates it with damped least-squares IK into absolute joint targets.

**How is an episode a success?**
The task predicate has to be true on the current state and stay true for 0.5 s. Earlier grasps and placements are not remembered, except the first-placement step that `in_order` compares. A lost object ends the episode as a failure.

**Which Gym ID should I start with?**
Train `Nepher-FrankaTabletopRL-LL-v0`, then play the planner with `Nepher-FrankaTabletopRL-HL-Play-v0`. For eval-nav, copy the checkpoint to `best_policy/best_policy.pt` and use `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0`.

**Which EnvHub bundle loads by default?**
`franka-tabletop-rl-v0` (16 tasks). Select another with `env_id`, for example `env_id=franka-tabletop-rl-v1`. Download it first: `nepher download <env_id>`.

**Where does the tracker checkpoint load from?**
`Nepher-FrankaTabletopRL-LL-Play-v0` and `Nepher-FrankaTabletopRL-HL-Play-v0` load the latest `model_*.pt` under `logs/rsl_rl/franka_tabletop_ll_ee_tracking` unless you pass `--checkpoint`. `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` loads `best_policy/best_policy.pt`.

## License

Copyright (c) 2026, [Nepher Robotics](https://github.com/nepher-ai). All rights reserved.
Distributed under the [BSD 3-Clause License](LICENSE).

**Keywords:** Franka, Robotiq 2F-85, pick-and-place, tabletop, reinforcement learning, PPO, hierarchical policy, inverse kinematics, LeRobot, NVIDIA Isaac Lab, Isaac Sim, rsl_rl, EnvHub, franka-tabletop-rl-v0, Nepher.
