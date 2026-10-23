# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene, physics, and asset wiring. Rates and sensors match ``task-franka-tabletop``."""

from __future__ import annotations

from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from franka_tabletop_rl.benchmark.stage import (
    read_stage,
    resolve_background,
    resolve_surface,
    surface_pose,
    xyzw_from_wxyz,
)
from franka_tabletop_rl.benchmark.stage_plan import SURFACE_ENTITY, ObjectVariant, StagePlan
from franka_tabletop_rl.scene.robot import franka_2f85, with_position_iterations
from franka_tabletop_rl.scene.sensors import (
    exterior_left_camera,
    exterior_right_camera,
    finger_contact,
    front_camera,
    wrist_camera,
)

CONTROL_HZ = 15
EPISODE_LENGTH_S = 40.0


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, table, and cameras. Object bodies are added from the stage plan."""

    robot = franka_2f85()
    # Mount under the arm. The benchmark replaces the mesh and the pose.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table_fixture",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/fixtures/franka_table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.087, 0.0, 0.0),
            # 180 degrees about +Z. The benchmark stores that turn as wxyz.
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
    )
    # Work surface the objects rest on. The benchmark chooses which table mesh.
    # Those payloads are dynamic rigid bodies. Hold the surface kinematic so the collider
    # stays on the visual top and placed objects do not fall through it.
    surface = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/fixtures/pending.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    cam_exterior_left = exterior_left_camera()
    cam_exterior_right = exterior_right_camera()
    cam_wrist = wrist_camera()
    cam_front = front_camera()
    gripper_left = finger_contact("left_inner_finger")
    gripper_right = finger_contact("right_inner_finger")
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.697)),
    )
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )
    background = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            texture_file="assets/backgrounds/home_office.exr",
            intensity=500.0,
            visible_in_primary_ray=True,
            texture_format="latlong",
        ),
    )


def apply_tabletop_physics(cfg) -> None:
    """120 Hz physics on a manager-based env cfg, so a command lands every 8 steps (15 Hz)."""
    cfg.episode_length_s = EPISODE_LENGTH_S
    set_physics_rate(cfg, 120)
    cfg.sim.physics = PhysxAutoCfg(isaacsim_physx=PhysxCfg())
    cfg.sim.physics.isaacsim_physx.enable_enhanced_determinism = True
    # GPU dynamics has no CCD, so the enableCCD authored on the pads and objects is inert.
    cfg.sim.physics.isaacsim_physx.enable_ccd = False
    # The default patch buffer is 163840. A full tabletop stage asks for far more.
    physx = cfg.sim.physics.isaacsim_physx
    physx.gpu_max_rigid_patch_count = 2**21
    physx.gpu_found_lost_pairs_capacity = 2**23


def set_physics_rate(cfg, rate_hz: int) -> None:
    """Step physics at ``rate_hz`` [Hz] and keep the control rate at :data:`CONTROL_HZ`.

    Raises:
        ValueError: ``rate_hz`` is not a positive multiple of the control rate.
    """
    if rate_hz < CONTROL_HZ or rate_hz % CONTROL_HZ:
        raise ValueError(f"physics rate_hz must be a multiple of {CONTROL_HZ}, got {rate_hz}")
    cfg.decimation = rate_hz // CONTROL_HZ
    cfg.sim.dt = 1.0 / rate_hz
    cfg.sim.render_interval = cfg.decimation


def apply_benchmark_physics(cfg, physics: dict) -> None:
    """Apply ``rate_hz`` [Hz], solver iterations, determinism, and patch count. Absent keys stay."""
    if "gpu_max_rigid_patch_count" in physics:
        cfg.sim.physics.isaacsim_physx.gpu_max_rigid_patch_count = int(physics["gpu_max_rigid_patch_count"])
    if "rate_hz" in physics:
        set_physics_rate(cfg, int(physics["rate_hz"]))
    if "solver_position_iterations" in physics:
        cfg.scene.robot = with_position_iterations(cfg.scene.robot, int(physics["solver_position_iterations"]))
    if "enhanced_determinism" in physics:
        cfg.sim.physics.isaacsim_physx.enable_enhanced_determinism = bool(physics["enhanced_determinism"])


def bundle_file(bundle: Path, relative: str) -> str:
    path = bundle / relative
    if not path.is_file():
        raise FileNotFoundError(f"benchmark is missing {relative} under {bundle}")
    return str(path)


def apply_stage_assets(cfg, bundle: Path, stage: dict | None = None, background: str | None = None) -> dict:
    """Point the dome, table, light, and ground at the stage. ``background`` overrides the dome.

    A surface that is not a multi-asset spawner keeps the stage default mesh.
    """
    if stage is None:
        stage = read_stage(bundle)
    apply_benchmark_physics(cfg, dict(stage.get("physics") or {}))
    dome = resolve_background(stage, background)
    cfg.scene.background.spawn = cfg.scene.background.spawn.replace(
        texture_file=bundle_file(bundle, dome["texture"]),
        intensity=float(dome.get("intensity", 500.0)),
        visible_in_primary_ray=bool(dome.get("visible_in_primary_ray", True)),
        texture_format=str(dome.get("texture_format", "latlong")),
    )
    fixture = stage["fixture"]
    cfg.scene.table.spawn = cfg.scene.table.spawn.replace(usd_path=bundle_file(bundle, fixture["usd"]))
    cfg.scene.table.init_state = cfg.scene.table.init_state.replace(
        pos=tuple(fixture.get("pos", (0.0, 0.0, 0.0))),
        rot=xyzw_from_wxyz(tuple(fixture.get("rot", (1.0, 0.0, 0.0, 0.0)))),
    )
    surface_cfg = getattr(cfg.scene, "surface", None)
    if surface_cfg is not None and not isinstance(surface_cfg.spawn, sim_utils.MultiAssetSpawnerCfg):
        surface = resolve_surface(stage, None)
        surface_cfg.spawn = surface_cfg.spawn.replace(usd_path=bundle_file(bundle, surface["usd"]))
        pos, rot = surface_pose(surface, {})
        surface_cfg.init_state = surface_cfg.init_state.replace(pos=pos, rot=xyzw_from_wxyz(rot))
    if "light" in stage:
        light = stage["light"]
        cfg.scene.light.spawn = cfg.scene.light.spawn.replace(intensity=float(light["intensity"]))
        cfg.scene.light.init_state = cfg.scene.light.init_state.replace(pos=tuple(light["pos"]))
    if "ground" in stage:
        ground = stage["ground"]
        cfg.scene.plane.spawn = cfg.scene.plane.spawn.replace(visible=bool(ground.get("visible", False)))
        cfg.scene.plane.init_state = cfg.scene.plane.init_state.replace(pos=tuple(ground["pos"]))
    return stage


def mount_stage(cfg, bundle: Path, stage: dict, plan: StagePlan) -> None:
    """Mount one body per object position and work surface. Build inside ``planned_worlds``."""
    for index, name in enumerate(plan.entities):
        position, quat = plan.start_poses[index]
        spawns = [_variant_spawn(bundle, variant) for variant in plan.variants[index]]
        setattr(cfg.scene, name, _body(name, position, quat, spawns))
    for index, name in enumerate(plan.surface_entities):
        variants = []
        for surface_name in plan.surface_variants[index]:
            if not surface_name:
                variants.append(_parked_spawn())
                continue
            record = resolve_surface(stage, surface_name)
            variants.append(cfg.scene.surface.spawn.replace(usd_path=bundle_file(bundle, str(record["usd"]))))
        pose = plan.surface_pose_by_phase[0][0]
        prim_path = cfg.scene.surface.prim_path if name == SURFACE_ENTITY else None
        body = _body(name, pose[0], pose[1], variants, prim_path=prim_path)
        setattr(cfg.scene, name, body)
    if SURFACE_ENTITY not in plan.surface_entities:
        cfg.scene.surface = None
    filters = ["{ENV_REGEX_NS}/" + name for name in plan.entities]
    cfg.scene.gripper_left.filter_prim_paths_expr = filters
    cfg.scene.gripper_right.filter_prim_paths_expr = filters
    # Every env owns its meshes, so physics cannot copy env 0. planned_worlds supplies the worlds.
    cfg.scene.replicate_physics = False
    cfg.scene.clone_cfg = CloneCfg(
        clone_combinations=[InclusionSet(assets=[*plan.entities, *plan.surface_entities])]
    )


def _body(name: str, position, quat, assets: list, prim_path: str | None = None) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path or "{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=xyzw_from_wxyz(quat)),
    )


def _tune_spawn(spawn: sim_utils.UsdFileCfg, scale, kinematic: bool) -> sim_utils.UsdFileCfg:
    if scale:
        spawn = spawn.replace(scale=tuple(float(value) for value in scale))
    if kinematic:
        spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
    return spawn


def _parked_spawn() -> sim_utils.CuboidCfg:
    """Stand-in for an unused position: a small kinematic cube that is parked out of the way."""
    return sim_utils.CuboidCfg(
        size=(0.02, 0.02, 0.02),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    )


def _variant_spawn(bundle: Path, variant: ObjectVariant | None):
    if variant is None:
        return _parked_spawn()
    spawn = sim_utils.UsdFileCfg(
        usd_path=bundle_file(bundle, variant.usd),
        activate_contact_sensors=True,
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    return _tune_spawn(spawn, variant.scale, variant.kinematic)


def read_stage_file(bundle: Path) -> dict:
    return read_stage(bundle)


# Re-exported so a task file can still ask for one surface pose while mounting.
__all__ = [
    "CONTROL_HZ",
    "EPISODE_LENGTH_S",
    "TabletopSceneCfg",
    "apply_benchmark_physics",
    "apply_stage_assets",
    "apply_tabletop_physics",
    "bundle_file",
    "mount_stage",
    "set_physics_rate",
]
