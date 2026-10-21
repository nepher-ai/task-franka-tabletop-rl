# Vendored identity layer

Everything in this directory defines the robot, the physics and the scene of the tabletop benchmark.
It is the code of `task-franka-tabletop` (`source/franka_tabletop/franka_tabletop`), commit
`4d64971edd9474e5700fbd3a196a4f85473aad3e`, copied so this project does not depend on the brain-runtime package.

The directory layout is the original one, so every relative import resolves unchanged.
The files below are byte-identical copies. `tests/test_parity.py` checks their sha256 values and
compares the configs built from them with the configs of an installed `franka_tabletop`.

| File | sha256 |
| --- | --- |
| `__init__.py` | `4d9f4400fcac2f4c2f010a1f806cd880a78f42b5966b855f561485727ae85ba8` |
| `assets/robots/franka_robotiq_2f85.usd` | `8f6a98811686863f4efe93968178673e1f8d7ccc1d50ae7050010fbc7aa9f1d8` |
| `core/__init__.py` | `eae38e49552e9d8f6438da514563eebb96c37b001a38cab50d2f40108e0d4890` |
| `core/clone.py` | `8a0049669eeaa0599b8abb747aed20ee20a563207cc9e7d08d7f9af11f056125` |
| `core/contact.py` | `4082e8a3b9e96984650f940b67b0e7c97d0a48fea639275f69a53638c4e6e689` |
| `core/observation_spec.py` | `6fe87e5d222e802defb02d0d4f5481731465785f1e30f843eb044a3649f0ba97` |
| `core/observations.py` | `434e6b2df8ac9cfaa57454e6fdcd1ceeb310767606aa26143101b027cd69a179` |
| `core/robot.py` | `554acbbde3938bcbc904f67ca5cdf7379e21d51059a35532ae00d362ccd055d4` |
| `core/sensors.py` | `ad897057d15cab0512b14ad0b45ff912accf1201a7fc14f557f60ca816127c8d` |
| `instructions.py` | `d11c6cd9816d0a5ff701e5d1f030ce298790209fd5daf96586bdeb63e5082f83` |
| `paths.py` | `aab8b6d5d0b7fd223727232e92690ed87cafa2cc89ceef038a4a1a9df36c25b8` |
| `predicates.py` | `3b74d0a4626a3c5017c443fc007a4169b332314447ce90d25df881550a30068f` |
| `scenes/__init__.py` | `41265299eb58a603b90a626b541bbfd5ae7eed2bc402f2830176c3435056d98e` |
| `scenes/bundle.py` | `f9247f62e2d02cf7eea547c36de0538378ed63f7eddda4e6c061ec88a6a71855` |
| `scenes/catalog.py` | `e7b39ce60d5790c8cc13ba5d0b2d183d710cc17cbee64e1508e70c8a8a795ff5` |
| `scenes/mixed.py` | `71ffa89fd7dd20bc0004daac18e44747a92e4009963a9bf8d676fe1a3ee80fd6` |
| `scenes/composer.py` | `5d00b46bb377fa73321e323cbb673b10344e0a1e544850e8734db0e384be58da` |
| `spec.py` | `79e4f502f2ba4eeec73717c20f01fe7b7bad96eead0619925e649d7119b7c894` |

Three files are written for this project and are not copies:

- `scene_cfg.py` is the identity part of `franka_tabletop/env_cfg.py`: `TabletopSceneCfg`, the physics rate
  and buffers, the bundle wiring, and `mount_mixed` (one stage that holds a different task per env).
  Brain-only pieces (observation narrowing, video clips) are left out.
  Parity is tested field by field, not by hash.
- `tiling.py` repeats a mixed-stage plan so that env `i` runs scene `i % T`.
- `bundle_path.py` locates the EnvHub bundle in the nepher cache.

To update the vendored files, copy the new originals, regenerate the table above, and run the parity test.
