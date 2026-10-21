# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO settings of the low-level tracker."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from franka_tabletop_rl.paths import best_policy_path, log_dir


@configclass
class LLPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner. The policy is reactive, so a small MLP with a full Markov observation is enough."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    # Absolute path. Isaac Lab joins this onto ``logs/rsl_rl``; an absolute
    # component replaces that prefix, so runs land in the task project's logs.
    experiment_name = str(log_dir())
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LLPPORunnerCfg_ENVHUB_PLAY(LLPPORunnerCfg):
    """Benchmark play loads the published tracker, not the newest training log."""

    load_checkpoint = str(best_policy_path())
