# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robot, physics and scene of this project must equal those of ``franka_tabletop``.

Hash checks run anywhere. Config checks need Isaac Lab and an installed ``franka_tabletop``,
and run with the Isaac Lab python, for example ``isaaclab.bat -p -m pytest tests/test_parity.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source" / "franka_tabletop_rl"
VENDORED = PACKAGE / "franka_tabletop_rl" / "tabletop"
sys.path.insert(0, str(PACKAGE))


def _digests() -> dict[str, str]:
    table = {}
    for line in (VENDORED / "VENDORED.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", line)
        if match:
            table[match.group(1)] = match.group(2)
    return table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_vendored_files_match_the_recorded_hashes():
    table = _digests()
    assert "assets/robots/franka_robotiq_2f85.usd" in table
    assert "core/robot.py" in table
    for relative, expected in table.items():
        assert _sha256(VENDORED / relative) == expected, f"{relative} drifted from the recorded copy"


def test_every_vendored_file_is_recorded():
    table = _digests()
    own = {"scene_cfg.py", "bundle_path.py", "tiling.py", "VENDORED.md"}
    present = {
        path.relative_to(VENDORED).as_posix()
        for path in VENDORED.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert present - own == set(table)


def test_vendored_files_match_the_original_package():
    original = pytest.importorskip("franka_tabletop")
    source = Path(original.__file__).parent
    for relative in _digests():
        assert _sha256(VENDORED / relative) == _sha256(source / relative), f"{relative} differs from franka_tabletop"


def _plain(value):
    """Recursive dict form of a configclass that ignores object identities."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return value


def _strip_paths(value):
    """Asset paths differ by design: each package ships its own copy of the robot USD."""
    if isinstance(value, dict):
        return {
            k: ("<asset>" if k == "usd_path" and isinstance(v, str) and v.endswith(".usd") else _strip_paths(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


@pytest.fixture(scope="module")
def pair():
    pytest.importorskip("franka_tabletop")
    pytest.importorskip("isaaclab.sim")
    from franka_tabletop import env_cfg as original
    from franka_tabletop_rl.tabletop import scene_cfg as vendored

    return original, vendored


def test_robot_cfg_matches(pair):
    from franka_tabletop.core import robot as original
    from franka_tabletop_rl.tabletop.core import robot as vendored

    left = _strip_paths(_plain(original.franka_2f85()))
    right = _strip_paths(_plain(vendored.franka_2f85()))
    assert left == right
    assert original.READY_JOINT_POS == vendored.READY_JOINT_POS
    assert original.ACTION_JOINTS == vendored.ACTION_JOINTS
    assert _plain(original.joint_position_action()) == _plain(vendored.joint_position_action())


def test_scene_cfg_matches(pair):
    original, vendored = pair
    left = _strip_paths(_plain(original.TabletopSceneCfg(num_envs=1, env_spacing=2.5)))
    right = _strip_paths(_plain(vendored.TabletopSceneCfg(num_envs=1, env_spacing=2.5)))
    assert left == right


def test_physics_matches(pair):
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.utils import configclass

    original, vendored = pair

    @configclass
    class Probe(ManagerBasedRLEnvCfg):
        def __post_init__(self):
            vendored.apply_tabletop_physics(self)

    reference = original.TabletopBenchmarkEnvCfg()
    probe = Probe()
    assert probe.decimation == reference.decimation == 8
    assert probe.sim.dt == reference.sim.dt
    assert probe.sim.render_interval == reference.sim.render_interval
    assert probe.episode_length_s == reference.episode_length_s == 40.0
    assert _plain(probe.sim.physics) == _plain(reference.sim.physics)
    assert vendored.CONTROL_HZ == original.CONTROL_HZ == 15


def test_stage_and_object_wiring_matches(pair, tmp_path):
    """The same bundle gives the same background, table, surface and object bodies."""
    import json

    from franka_tabletop_rl.tabletop.bundle_path import resolve_bundle

    original, vendored = pair
    try:
        bundle = resolve_bundle()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    scene = sorted((bundle / "scenes").glob("*.json"))[0]
    task_id, scene_id = scene.stem.split("-", 1)
    reference = original.TabletopBenchmarkEnvCfg()
    reference.task_id, reference.scene_id, reference.bundle_path = task_id, scene_id, str(bundle)
    original._apply_benchmark_assets(reference)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.utils import configclass

    @configclass
    class Probe(ManagerBasedRLEnvCfg):
        scene: vendored.TabletopSceneCfg = vendored.TabletopSceneCfg(num_envs=1, env_spacing=2.5)

        def __post_init__(self):
            vendored.apply_tabletop_physics(self)

    probe = Probe()
    payload = json.loads(scene.read_text(encoding="utf-8"))
    vendored.apply_stage_assets(probe, bundle, payload)
    slots = vendored.mount_objects(probe, bundle, payload["placements"], scene.stem)
    assert slots == reference.object_slots
    for name in ("background", "table", "surface", "light", "plane", *slots):
        assert _plain(getattr(probe.scene, name)) == _plain(getattr(reference.scene, name)), name
    assert _plain(probe.scene.gripper_left) == _plain(reference.scene.gripper_left)
    assert _strip_paths(_plain(probe.scene.robot)) == _strip_paths(_plain(reference.scene.robot))
    assert probe.sim.dt == reference.sim.dt


def _variants(plan) -> list:
    """Variants as plain tuples, since each package has its own ``ObjectVariant`` class."""
    return [[None if v is None else dataclasses.astuple(v) for v in column] for column in plan.variants]


def test_mixed_stage_matches(pair):
    """Several tasks in one stage mount the same bodies, meshes and surface as the original."""
    import json

    from franka_tabletop_rl.tabletop.bundle_path import resolve_bundle
    from franka_tabletop_rl.tabletop.scenes.bundle import read_stage

    original, vendored = pair
    try:
        bundle = resolve_bundle()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    files = sorted((bundle / "scenes").glob("*.json"))[:3]
    pairs = tuple(tuple(path.stem.split("-", 1)) for path in files)
    reference = original.TabletopBenchmarkEnvCfg()
    reference.task_id, reference.scene_id = pairs[0]
    reference.bundle_path = str(bundle)
    reference.mixed_scenes = pairs
    reference.scene.num_envs = len(pairs)
    original._apply_benchmark_assets(reference)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.utils import configclass

    @configclass
    class Probe(ManagerBasedRLEnvCfg):
        scene: vendored.TabletopSceneCfg = vendored.TabletopSceneCfg(num_envs=len(pairs), env_spacing=2.5)

        def __post_init__(self):
            vendored.apply_tabletop_physics(self)

    probe = Probe()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    vendored.apply_stage_assets(probe, bundle, payloads[0])
    plan = vendored.mount_mixed(probe, bundle, read_stage(bundle), payloads)
    expected = reference.mixed_plan
    assert plan.entities == expected.entities
    assert _variants(plan) == _variants(expected)
    assert plan.choices == expected.choices
    assert plan.slot_entities == expected.slot_entities
    assert plan.surface_poses == expected.surface_poses
    for name in (*plan.entities, "surface"):
        assert _plain(getattr(probe.scene, name)) == _plain(getattr(reference.scene, name)), name
    assert _plain(probe.scene.gripper_left) == _plain(reference.scene.gripper_left)
    assert probe.scene.replicate_physics == reference.scene.replicate_physics


def test_tile_plan_repeats_tasks_in_env_order():
    from franka_tabletop_rl.tabletop.scenes.mixed import plan_mixed
    from franka_tabletop_rl.tabletop.tiling import tile_plan

    stage = {"backgrounds": [{"name": "bg"}], "surfaces": [{"name": "oak", "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}]}

    def scene(usd: str) -> dict:
        placement = {"slot": "a", "kind": "item", "usd": usd, "position_m": [0.5, 0.0, 0.0]}
        return {"background": "bg", "surface": "oak", "placements": [placement]}

    plan = plan_mixed([scene("x.usd"), scene("y.usd")], stage)
    tiled = tile_plan(plan, 6)
    assert tiled.num_envs == 6
    assert tiled.choices[0] == (0, 1, 0, 1, 0, 1)
    assert tiled.surface_choices == (0, 0, 0, 0, 0, 0)
    assert tiled.slot_entities[3] == plan.slot_entities[1]
    with pytest.raises(ValueError):
        tile_plan(plan, 5)
