# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene, physics and asset wiring of the tabletop benchmark.

This module holds the parts of ``franka_tabletop.env_cfg`` that define the identity of the
environment: the scene entities, the physics rate and buffers, and the way a benchmark bundle
fills the table, surface, background and object meshes. The code is the original code with the
brain-only pieces (observation narrowing, video clips) left out.
``tests/test_parity.py`` compares every field against ``franka_tabletop``.
"""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .core.robot import franka_2f85, with_position_iterations
from .core.sensors import exterior_left_camera, exterior_right_camera, finger_contact, front_camera, wrist_camera
from .scenes.bundle import read_stage, resolve_background, resolve_surface, surface_pose, xyzw_from_wxyz
from .scenes.mixed import SURFACE_ENTITY, MixedPlan, ObjectVariant, plan_mixed

# Replaced from the benchmark scene before the simulation starts.
_PENDING_USD = "assets/objects/pending.usd"
# The brain is commanded at this rate whatever physics rate the benchmark declares.
CONTROL_HZ = 15
EPISODE_LENGTH_S = 40.0


def _object(
    slot: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> RigidObjectCfg:
    """One contact slot. The benchmark scene supplies the mesh, including its rigid body.

    ``rotation`` is xyzw. Reset settles from this pose before the episode writes
    positions again, so a yawed fixture must not spawn at identity.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + slot,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PENDING_USD,
            activate_contact_sensors=True,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation),
    )


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Robot, table, and cameras. Object bodies are added from the scene file."""

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
    """Physics of the tabletop benchmark: 120 Hz steps, a command every 8 steps (15 Hz).

    ``cfg`` is any manager-based env cfg. One control step is 1/15 s.
    """
    cfg.episode_length_s = EPISODE_LENGTH_S
    set_physics_rate(cfg, 120)
    cfg.sim.physics = PhysxAutoCfg(isaacsim_physx=PhysxCfg())
    cfg.sim.physics.isaacsim_physx.enable_enhanced_determinism = True
    # GPU dynamics has no CCD, so the enableCCD authored on the pads and objects is inert.
    cfg.sim.physics.isaacsim_physx.enable_ccd = False
    # Sixteen envs share one PhysX scene. The default patch buffer is 163840, and a
    # 7-env tabletop already asked for about 194000. Both pair buffers are sized for a full shard.
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
    """Apply the stage's optional ``physics`` block. Absent keys keep the task default.

    Keys: ``rate_hz`` [Hz], ``solver_position_iterations`` for the robot,
    ``enhanced_determinism``, and ``gpu_max_rigid_patch_count`` for stages with many envs.
    """
    if "gpu_max_rigid_patch_count" in physics:
        cfg.sim.physics.isaacsim_physx.gpu_max_rigid_patch_count = int(physics["gpu_max_rigid_patch_count"])
    if "rate_hz" in physics:
        set_physics_rate(cfg, int(physics["rate_hz"]))
    if "solver_position_iterations" in physics:
        cfg.scene.robot = with_position_iterations(cfg.scene.robot, int(physics["solver_position_iterations"]))
    if "enhanced_determinism" in physics:
        cfg.sim.physics.isaacsim_physx.enable_enhanced_determinism = bool(physics["enhanced_determinism"])


def _choice(payload: dict, key: str) -> str | None:
    """Scene override for a stage list. Absent means the stage default."""
    value = payload.get(key)
    return value if isinstance(value, str) else None


def bundle_file(bundle: Path, relative: str) -> str:
    """Absolute path of one file inside a downloaded benchmark."""
    path = bundle / relative
    if not path.is_file():
        raise FileNotFoundError(f"benchmark is missing {relative} under {bundle}")
    return str(path)


def read_scene_payload(bundle: Path, task_id: str, scene_id: str) -> dict:
    """Raw scene JSON of one task. Raises ``FileNotFoundError`` when the bundle has no such scene."""
    scene_file = bundle / "scenes" / f"{task_id}-{scene_id}.json"
    if not scene_file.is_file():
        raise FileNotFoundError(f"benchmark has no scene {task_id}-{scene_id} under {bundle}")
    return json.loads(scene_file.read_text(encoding="utf-8"))


def apply_stage_assets(cfg, bundle: Path, payload: dict | None = None) -> dict:
    """Point the office, table, surface, light and ground at what the stage declares.

    Returns the stage dict. ``payload`` is a scene file; without it the stage defaults apply.
    """
    payload = payload or {}
    stage = read_stage(bundle)
    apply_benchmark_physics(cfg, dict(stage.get("physics") or {}))
    background = resolve_background(stage, _choice(payload, "background"))
    cfg.scene.background.spawn = cfg.scene.background.spawn.replace(
        texture_file=bundle_file(bundle, background["texture"]),
        intensity=float(background.get("intensity", 500.0)),
        visible_in_primary_ray=bool(background.get("visible_in_primary_ray", True)),
        texture_format=str(background.get("texture_format", "latlong")),
    )
    fixture = stage["fixture"]
    cfg.scene.table.spawn = cfg.scene.table.spawn.replace(usd_path=bundle_file(bundle, fixture["usd"]))
    cfg.scene.table.init_state = cfg.scene.table.init_state.replace(
        pos=tuple(fixture.get("pos", (0.0, 0.0, 0.0))),
        rot=xyzw_from_wxyz(tuple(fixture.get("rot", (1.0, 0.0, 0.0, 0.0)))),
    )
    surface = resolve_surface(stage, _choice(payload, "surface"))
    pos, rot = surface_pose(surface, payload)
    cfg.scene.surface.spawn = cfg.scene.surface.spawn.replace(usd_path=bundle_file(bundle, surface["usd"]))
    cfg.scene.surface.init_state = cfg.scene.surface.init_state.replace(pos=pos, rot=xyzw_from_wxyz(rot))
    if "light" in stage:
        light = stage["light"]
        cfg.scene.light.spawn = cfg.scene.light.spawn.replace(intensity=float(light["intensity"]))
        cfg.scene.light.init_state = cfg.scene.light.init_state.replace(pos=tuple(light["pos"]))
    if "ground" in stage:
        ground = stage["ground"]
        cfg.scene.plane.spawn = cfg.scene.plane.spawn.replace(visible=bool(ground.get("visible", False)))
        cfg.scene.plane.init_state = cfg.scene.plane.init_state.replace(pos=tuple(ground["pos"]))
    return stage


def _tune_spawn(spawn: sim_utils.UsdFileCfg, scale, kinematic: bool) -> sim_utils.UsdFileCfg:
    """Apply a placement's instance scale and kinematic flag to an object spawner."""
    # An instance scale shrinks an asset that is larger than the table. size_m is already
    # that scaled extent, so the score box matches the mesh that is spawned.
    if scale:
        spawn = spawn.replace(scale=tuple(float(value) for value in scale))
    # A shelf is a fixed collider in the source scene. A dynamic copy falls into the table.
    if kinematic:
        spawn.rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
    return spawn


def mount_objects(cfg, bundle: Path, placements: list[dict], label: str = "") -> tuple[str, ...]:
    """Spawn one rigid body per placement and point both finger filters at those prims.

    Returns the slot names in placement order, which is the finger filter order.
    """
    if not placements:
        raise ValueError(f"scene {label} places no objects")
    slots: list[str] = []
    filters: list[str] = []
    for item in placements:
        if str(item.get("kind")) == "anchor":
            continue
        slot = str(item["slot"])
        if slot in slots:
            raise ValueError(f"scene {label} places {slot} more than once")
        position = tuple(float(value) for value in item["position_m"])
        quat = tuple(float(value) for value in item.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
        entity = _object(slot, position, xyzw_from_wxyz(quat))
        entity.spawn = _tune_spawn(
            entity.spawn.replace(usd_path=bundle_file(bundle, str(item["usd"]))),
            item.get("scale"),
            bool(item.get("kinematic")),
        )
        setattr(cfg.scene, slot, entity)
        slots.append(slot)
        filters.append("{ENV_REGEX_NS}/" + slot)
    cfg.scene.gripper_left.filter_prim_paths_expr = filters
    cfg.scene.gripper_right.filter_prim_paths_expr = filters
    return tuple(slots)


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


def mount_mixed(cfg, bundle: Path, stage: dict, payloads: list[dict]) -> MixedPlan:
    """Mount one body per object position and one for the work surface, each with a mesh per env.

    The finger filters name the positions, so every env has the same filter count. Build the env
    inside :func:`~franka_tabletop_rl.tabletop.core.clone.planned_worlds` so the cloner gets one
    world per env.

    Args:
        cfg: Manager-based env cfg whose ``scene`` is a :class:`TabletopSceneCfg`.
        bundle: Directory of the benchmark bundle.
        stage: The bundle's ``assets/stage.json``.
        payloads: One scene file per env, in env order.

    Returns:
        The plan that maps every env to its meshes.
    """
    plan = plan_mixed(payloads, stage)
    for index, name in enumerate(plan.entities):
        position, quat = plan.start_poses[index]
        setattr(
            cfg.scene,
            name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                # The wrapper replaces each variant's contact-sensor flag, so it is set here.
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[_variant_spawn(bundle, variant) for variant in plan.variants[index]],
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=xyzw_from_wxyz(quat)),
            ),
        )
    surfaces = [resolve_surface(stage, name) for name in plan.surfaces]
    position, quat = plan.surface_poses[0]
    # The surface is a kinematic body, so each env can move it to its own scene's pose.
    setattr(
        cfg.scene,
        SURFACE_ENTITY,
        RigidObjectCfg(
            prim_path=cfg.scene.surface.prim_path,
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    cfg.scene.surface.spawn.replace(usd_path=bundle_file(bundle, str(record["usd"])))
                    for record in surfaces
                ],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=xyzw_from_wxyz(quat)),
        ),
    )
    filters = ["{ENV_REGEX_NS}/" + name for name in plan.entities]
    cfg.scene.gripper_left.filter_prim_paths_expr = filters
    cfg.scene.gripper_right.filter_prim_paths_expr = filters
    # Every env owns its meshes, so physics cannot copy env 0. The claim only keeps the cloner on
    # its heterogeneous path; planned_worlds supplies the actual worlds.
    cfg.scene.replicate_physics = False
    cfg.scene.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=[*plan.entities, SURFACE_ENTITY])])
    return plan
