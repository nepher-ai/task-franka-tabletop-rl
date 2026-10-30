# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw wireframe previews of a film shot from the benchmark layout. It needs no simulator or GPU.

It shows where each shot's camera points and how much of the table fits, so the lens can be tuned before a
render. The result is a PNG contact sheet at the camera key times and the cut points. The real framing is
confirmed with the preview mode of ``record_film.py``.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))
try:
    import franka_tabletop_rl.film  # noqa: F401
except ImportError:
    # The package registers Gym tasks and needs Isaac Lab. The film previews do not, so load its folders only.
    root = PACKAGE / "franka_tabletop_rl"
    for name, folder in (("franka_tabletop_rl", root), ("franka_tabletop_rl.tasks", root / "tasks")):
        stub = types.ModuleType(name)
        stub.__path__ = [str(folder)]
        sys.modules[name] = stub

from franka_tabletop_rl.benchmark.bundle import DEFAULT_ENV_ID, load_benchmark, resolve_bundle  # noqa: E402
from franka_tabletop_rl.film import framing, shots  # noqa: E402
from franka_tabletop_rl.film.render import save_png  # noqa: E402

parser = argparse.ArgumentParser(description="Wireframe previews of a film shot.")
parser.add_argument("shot", help="Shot name, or 'all'.")
parser.add_argument(
    "--task", default=None, help="Task id whose layout to draw. Defaults to the shot's first focus task."
)
parser.add_argument(
    "--times", type=float, nargs="*", default=None, help="Take times [s]. Defaults to key times and cut points."
)
parser.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
parser.add_argument("--output", default=None, help="Folder for the PNG files. Defaults to <project>/film/framing.")
args = parser.parse_args()


def _times(shot: shots.Shot) -> list[float]:
    pre = shot.pre_roll_s
    times = {key.t for key in shot.keys} | {pre, pre + shot.cut.after_s}
    if shot.cut.anchor in ("start", "first_close"):
        times.add(max(pre - shot.cut.before_s, 0.0))
    times.add(0.5 * (min(times) + max(times)))
    return sorted(round(t, 2) for t in times)


def _tool_spot(boxes: list[framing.Box]) -> np.ndarray:
    """Tool position [m]: just above the first object, as at the grasp."""
    objects = [box for box in boxes[1:]]
    if not objects:
        return np.array([0.45, 0.0, 0.26])
    target = objects[0]
    return target.center + (0.0, 0.0, target.half[2] + 0.02)


def _draw(shot: shots.Shot, benchmark) -> Path:
    task = args.task or (shot.focus[0] if shot.focus else benchmark.select("all")[0])
    layout = benchmark.task(task)
    boxes = framing.layout_boxes(layout, benchmark.stage)
    times = args.times if args.times else _times(shot)
    count = len(benchmark.select("all")) if shot.all_envs else 1
    frames = framing.shot_frames(shot, boxes, _tool_spot(boxes), times, tuple(args.size), count)
    out = Path(args.output) if args.output else PACKAGE.parents[1] / "film" / "framing"
    path = out / (f"{shot.name}_{task}.png" if args.task else f"{shot.name}.png")
    save_png(path, framing.contact_sheet(frames, columns=2))
    print(f"[INFO] {shot.name} ({task}): {len(frames)} views at {times} s -> {path}")
    return path


def main() -> None:
    benchmark = load_benchmark(resolve_bundle(DEFAULT_ENV_ID))
    names = sorted(shots.SHOTS) if args.shot == "all" else [args.shot]
    for name in names:
        _draw(shots.get_shot(name), benchmark)


if __name__ == "__main__":
    main()
