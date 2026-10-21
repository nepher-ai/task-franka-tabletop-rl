# task-franka-tabletop-rl

Hierarchical Franka pick-and-place on the `task-franka-tabletop` benchmark scene, for Isaac Sim 6.1.0 and
Isaac Lab 3.0. The install target and the `isaaclab.bat train` / `play` commands follow `task-go2-lidar-maze`.
The project is not a brain runtime: there are no sockets or brain images.
[PRINCIPLES.md](PRINCIPLES.md) holds the constraints it follows.

```
classical planner (hl_policy) -> learned end-effector tracker (ll_policy) -> damped-least-squares IK -> 8-D joint targets
```

- **High level.** `ClassicalTabletopPlanner` picks each required object with a top-down grasp that it searches on the
  object's mesh, carries it over the container, lets go, and retries a failed grasp. It reads the benchmark's own success
  predicate, so it stops when the task is done.
- **Low level.** An RSL-RL PPO policy tracks the planner's tool pose and gripper command. Its 7-D action is a tool-pose delta
  plus the gripper. `IkJointCommandAction` turns it into absolute joint targets, which is the action space of the brains.
- **Scene identity.** `franka_tabletop_rl/tabletop/` is a vendored copy of the robot, physics, sensors, scenes, and success
  predicates of `task-franka-tabletop`. `tests/test_parity.py` checks the copy against the original, so a recorded trajectory is valid training data for that benchmark.

## Gym tasks

| Id | Use |
| --- | --- |
| `Nepher-FrankaTabletopRL-LL-v0`, `-LL-Play-v0` | Train and play the tracker |
| `Nepher-FrankaTabletopRL-HL-Play-v0` | Planner plus tracker on every benchmark task in one stage. Loads the latest tracker checkpoint |
| `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` | The same, built by eval-nav from an EnvHub bundle. Loads `best_policy/best_policy.pt` |

Both high-level tasks load the benchmark from the nepher cache of `franka-tabletop-rl-v0`
(`nepher download franka-tabletop-rl-v0`). That bundle has the 16 tasks of the tabletop benchmark.

**One stage, all tasks.** The high-level env builds one stage with `T` tasks (16 here, read from `benchmark.yaml`, so 24 or 32
work too). Env `i` runs task `i % T`, so `--num_envs` must be a multiple of `T`. Every env holds one body per object
position (`obj<k>`, a multi-asset spawner that gives each task its own meshes) and a kinematic `surface`. Positions that a
task does not use are parked away from the table. The layout code is the `mixed` scene machinery of `task-franka-tabletop`
(`tabletop/scenes/mixed.py`, `tabletop/core/clone.py`), and `tabletop/tiling.py` repeats the `T` tasks over `num_envs` envs.
Because every env owns its meshes, the cloner builds one world per env (`replicate_physics=False`), which costs build time.

`isaaclab.bat train` and `play` discover these through the `isaaclab.tasks` entry point.

## Install

Isaac Sim has to run with its bundled Python, so every command goes through `isaaclab.bat` from the Isaac Lab root
(`f:\IsaacLab`). Inside that tree this package is `source\task-franka-tabletop-rl\source\franka_tabletop_rl`.

```bat
.\isaaclab.bat -p -m pip install -e source\task-franka-tabletop-rl\source\franka_tabletop_rl
.\isaaclab.bat -p -m pip install -e "source\task-franka-tabletop-rl\source\franka_tabletop_rl[record]"
```

The second line adds the LeRobot writer. TensorBoard logs are under this project, not the Isaac Lab `logs` directory.

## Train and play

Isaac Lab 3 does not have a `--headless` flag. `--viz none` runs without a window. Playback runs until Ctrl+C.
`--num_envs` for a high-level task must be a multiple of the task count. One task is `env.scene_id=<task_id>`.

```bat
.\isaaclab.bat train --task Nepher-FrankaTabletopRL-LL-v0 --num_envs 4096 --max_iterations 1500 --viz none
.\isaaclab.bat play --task Nepher-FrankaTabletopRL-LL-Play-v0 --num_envs 64 --viz none
.\isaaclab.bat play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --viz none
.\isaaclab.bat play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 4 env.scene_id=throw_away_apple --viz none
```

`Nepher-FrankaTabletopRL-LL-Play-v0` and `Nepher-FrankaTabletopRL-HL-Play-v0` load the latest `model_*.pt` under
`logs/rsl_rl/franka_tabletop_ll_ee_tracking` unless you pass `--checkpoint`. Copy the checkpoint eval-nav should load to
`best_policy/best_policy.pt`. `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` loads that file. The checkpoint is not committed.

### Highlight videos

`env.highlight=on` on `Nepher-FrankaTabletopRL-HL-Play-v0` or `Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0` adds video
recording and changes nothing else: the tasks, `--num_envs`, and layouts stay those of the same command. It writes
`best_policy/videos/highlight/<camera>.mp4` for `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, and
`cam_front`. Each is 1280x720, a 4x4 grid of 320x180 cells, and lasts one episode (`episode_length_s`, 40 s). The
grid shows up to 16 tasks of the stage in benchmark order. A stage with more than 16 tasks shows 16 sampled with the
seed. A stage with fewer leaves the remaining cells black. Only those envs are rendered, at the cell size.

```bat
.\isaaclab.bat play --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --viz none env.highlight=on
```

### Dataset for brains

`scripts/record_lerobot.py` runs the planner and the tracker, and keeps the successful episodes as a LeRobot v2.1 dataset
(`--cameras` records every camera as h264 video, and `--cameras cam_wrist cam_front` records those). It records all tasks at once, or one with `--task <task_id>`.
`--episodes` is the number of successful episodes kept per task, and `--num_envs` a multiple of the task count.
It also copies the tracker it used to `best_policy/best_policy.pt`.

```bat
.\isaaclab.bat -p source\task-franka-tabletop-rl\scripts\record_lerobot.py --episodes 8 --num_envs 32 --output datasets\tabletop_rl
.\isaaclab.bat -p source\task-franka-tabletop-rl\scripts\replay_check.py --dataset datasets\tabletop_rl --episode 0
```

| Feature | Shape | Content |
| --- | --- | --- |
| `observation.state` | 8 | 7 joint positions [rad], gripper closure 0..1 |
| `observation.eef_pose` | 7 | tool position [m] and xyzw quaternion in the brain's convention |
| `action` | 8 | 7 absolute joint targets [rad], gripper 0 (open) or 1 (closed) |
| `observation.images.<camera>` | H, W, 3 | `cam_exterior_left`, `cam_exterior_right`, `cam_wrist`, `cam_front` |

`meta/modality.json` declares the GR00T embodiment `franka_2f85_jointpos`. `scripts/replay_check.py` replays an episode
recorded with `--jitter 0` in `task-franka-tabletop`'s own `TabletopEvalEnv`.

### eval-nav

`eval-nav/configs/task-franka-tabletop-rl.yaml` evaluates all 16 tasks in process with the single `env_scene` `all`: one stage,
`num_envs: 64` envs (four per task) and `num_episodes: 4` runs, so 16 jittered episodes of each task. The scorer
`manipulation.multitask.rl` (`episode = success * (0.75 + 0.25 * speed)`) maps episode `n` to task `n % T` of the bundle's
`benchmark.yaml`, so keep `all` as the only `env_scene`. On Linux, eval-nav forks one worker per scene. On Windows it
runs the scenes in the process that already started Kit. The task module's `tasks_of_scene` tells eval-nav which scene ids it owns and which tasks each
one runs, so eval-nav has no scene name of its own. The in-process runner and the `v11` to `v13` configs are untouched.

## Results

Measured on one RTX 4090 with the committed code and the checkpoint of iteration 1499.

- **Tracker.** The median position error is 0.10 cm and the median orientation error is 0.003 rad.
- **Planner plus tracker.** One stage with the 16 tasks and 64 envs (four per task), 4 cm layout jitter, 16 episodes of
  each task through the eval-nav worker steps (see below). The counts are episodes that succeeded:

| Task | Success | Task | Success |
| --- | --- | --- | --- |
| `animals_in_bin` | 12/16 | `red_dishes_in_bin` | 9/16 |
| `apple_and_yogurt_in_bowl` | 12/16 | `red_items_in_bin` | 11/16 |
| `bigger_object` | 11/16 | `rubiks_cubes_in_bin` | 16/16 |
| `blocks_in_bin` | 7/16 | `smaller_object` | 16/16 |
| `clamp_in_right_bin` | 16/16 | `throw_away_apple` | 16/16 |
| `cubes_and_blocks_in_bin` | 0/16 | `throw_away_snacks` | 16/16 |
| `foodpacking_1bin_1can` | 13/16 | `tool_organization` | 9/16 |
| `foodpacking_1bin_3can` | 0/16 | `wood_spatula_to_bowl` | 6/16 |

  That is 170 of 256 episodes. `cubes_and_blocks_in_bin` asks for seven objects in 40 s, which the planner's pace does not
  reach, and `foodpacking_1bin_3can` needs three cans in 40 s. The rates of the harder tasks move a lot between runs of
  16 episodes: `foodpacking_1bin_3can` succeeded in 7/16, 6/16, and 2/16 episodes in other runs, and a 64-env run
  with 8 episodes of each task gave 84 of 128 episodes in total.
- **Equivalence.** Two recorded episodes of `throw_away_apple` replayed in `TabletopEvalEnv` finish the task, and the joints
  follow the recorded path within 0.018 rad.
- **eval-nav.** On Windows, where `evaluate.py` cannot fork, a script that follows the in-process worker steps of eval-nav
  (config class, `gym.make`, `wrap_for_eval`, checkpoint policy) ran `task-franka-tabletop-rl.yaml` with `scene: all`: 64 envs,
  4 runs, 256 episodes in 13 minutes on one RTX 4090. The scorer's task groups held 16 episodes each. Building the
  64-env stage takes about 20 s, the same as 16 envs. `record_lerobot.py --num_envs 32` saved one episode of every task in 6 minutes.

## Tests

From this directory, `..\..\isaaclab.bat -p -m pytest tests` (no GPU needed): parity of the vendored layer, grasp search and the planner's
stage logic on a fake tracker, the eval-nav wrapper, and the LeRobot writer.
