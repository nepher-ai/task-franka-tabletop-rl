# task-franka-tabletop-rl — principles

These principles govern the Isaac Lab external project `task-franka-tabletop-rl`. They are constraints on the project. They are not a design, a file layout, or an implementation plan.

A Franka robot performs object manipulation on a table. Pick-and-place is the task the project must execute.

## Project shape

`task-franka-tabletop-rl` is a manager-based Isaac Lab external project, in the same family as `task-franka-pickandplace-multibase`, `task-humanoid-run-jump` and `task-go2-lidar-maze`. Mainly, `task-go2-lidar-maze`. Because, `task-franka-tabletop-rl` should be ran in IsaacSim 6.1.0/Isaaclab 3.0.0.

It is a trainable hierarchical policy project: registered Gym tasks, low-level and high-level policy configs, and the usual train and play entry points of that family.

It does not follow the brain-runtime paradigm used by `task-franka-tabletop`. Brain sockets, brain images, and tournament brain submissions are outside this project.

## Control stack

Pick-and-place uses the same policy split and the same command path as `task-franka-pickandplace-multibase`:

```
High-Level Planner (classical pick-and-place)
    → Low-Level Policy (ll_policy)
    → Inverse Kinematics (IK)
    → Joint Commands
```

`hl_policy` is the classical pick-and-place planner. `ll_policy` is the learned low-level policy that the planner commands. IK turns the low-level policy output into joint commands for the Franka.

Both policies follow that project's design approach. The high-level layer stays a classical planner. The low-level layer stays a policy trained to track the planner's commands through IK.

## Scene identity

The robot, the physics, and the environment configuration are identical to `task-franka-tabletop`.

That identity is what makes trajectories from this project valid training data for brains such as Pi0.5 and GR00T. The arm, the gripper, the table, the objects, the sensors, and the physics that those brains will later see are the tabletop project's, unchanged.

Policy structure may differ from `task-franka-tabletop`. The scene those policies act in may not.

## EnvHub and eval-nav

This project owns `franka-tabletop-rl-v0` (the original 16 tasks) and `franka-tabletop-rl-v1` (all 120 tasks, derived from the `franka-tabletop-v13` data). Both are local EnvHub bundles. The brain-runtime bundles `franka-tabletop-v11` through `franka-tabletop-v13`, and their eval-nav configs, stay untouched.

`eval-nav/configs/task-franka-tabletop-rl.yaml` evaluates v0. `eval-nav/configs/task-franka-tabletop-rl-v1.yaml` evaluates v1.
