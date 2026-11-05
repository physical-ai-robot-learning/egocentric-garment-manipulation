# tag_rig conversion on RunPod — dependency inventory (11-02)

Reference: `scratchpad/fl1102c_chain.sh` (steps 0–7) and `scratchpad/flat_build_upload.py` (LeRobot build, which stays on the Mac).
Everything here was traced read-only from the Mac tree on 2026-11-02: an AST import walk plus import tests in both venvs.
Nothing was run on a pod, so every pod-side statement is **untested**.

## 1. Steps, scripts, envs

Abbreviations: `E=~/ego_collector`, `IC=$E/datasets/m3_runs/_imu_consistency`, `B=$IC/orbbec/v2b`, `F=$E/datasets/m3_runs/_fgo_1023/tools`, `L=~/laundry_dit_training`, `TR=$E/datasets/tag_rig`.

| step | script (cwd) | Mac env → pod env | reads | writes |
|---|---|---|---|---|
| 0 servo jaw | `$E/scripts/jaw_from_motor.py` | ego/.venv → `envs/ego` | `TR/<t>/gripper_motor_{left,right}.npz` | `TR/<t>/{wrist,wrist_left}/umi_jaw.npz` |
| 1 tag traj | `-m laundry_tracking.calib.umi_tcp_traj --hand 오른손/왼손 --rot imu --imu-fill --robot-cal configs/calibration/robot_cam_v005.json --rig configs/calibration/umi_tag_rig_v002.json` (cwd `$E`) | ego | `TR/<t>/{meta.json,t_ns.npy,color_*.jpg,depth_*.png,imu_*.npz}`, `configs/calibration/umi_tag_rig_v002.json`, `robot_cam_v005.json` | `TR/<t>/umi_traj_{left,right}.npz`, `TR/<t>/segments.json` |
| 2 prep | `$L/datasets/tag_rig_to_lerobot.py --prep-only` (cwd `$L`) | xvla-mac → `envs/ik` | `TR/<t>/{wrist,wrist_left}/{t_ns.npy,wrist_*.jpg}` (optical-flow cam↔rig pairing, every 3rd frame), `umi_traj_*`, `segments.json`, `configs/calibration/jaw_kpt_calib_v002.json` (threshold check) | `$L/data/tag_rig_derived/<t>/{assignment.json,segments.json,umi_traj_{left,right}.npz,ik_v1/,robotize_ik.log}`. **Spawns** `~/xvla-mac/bin/python datasets/robotize_ik.py` with 6 workers (hard-coded `IK_WORKERS=6`). |
| 3a trims+seed | `-m laundry_tracking.new_take_prep <t>` | ego | `TR/<t>/{meta,umi_traj_*,imu_*}`, `data/tag_rig_derived/<t>/assignment.json`, `laundry_tracking/body_offset_by_rig_v1.json` (package resource) | **appends** to `$L/reports/idle_trim.csv` (+ `.csv.bak_1026`); writes `$B/start_seed_<t>.json` |
| 3b tag rot | `-m laundry_tracking.tag_rotations $TR/<t>` (reads `$E/configs/laundry/calibration/umi_tag_rig_v002.json`: aliased to `configs/calibration` in the job HOME) | ego | `color_*.jpg`, `t_ns.npy`, `umi_tag_rig_v002.json` | `$E/datasets/m3_runs/_fgo_1023/tagrot/<t>.npz` (the directory must exist) |
| 3c tracker | `-m laundry_tracking.orbbec_tracker <t>`, env `V31_P='{"JUMP":0.10,"V_GATE":1.0}' V31_OUT=trackv31speed` | ego | `color_*/depth_*`, `$B/start_seed_v1.json`, `start_seed_<t>.json`, idle_trim.csv, tagrot npz, `umi_tag_rig_v001.json`, `imu_*.npz` | `$B/out/trackv31speed_<t>.npz` |
| 4a pack | `-m laundry_tracking.cotrack.make_pack <pack> t:d ...` | ego | `color_*`/`depth_*` in the trim window, `out/trackv31speed_<t>.npz`, `robot_cam_v00*.json` (camcal) | `<pack>/<t>_dNN/{frames/*.jpg,meta.json}` (~25 MB per demo) |
| 4b CoTracker | `-m laundry_tracking.cotrack.run_cotracker <pack>` (CUDA, `.cuda()` hard-coded) / `run_cotracker_mps.py` (auto cuda/mps/cpu) | 4090 `~/ldt_tagrig/.venv` → `envs/ct` | `<pack>/*/frames`, `meta.json`; `torch.hub` `facebookresearch/co-tracker` + `scaled_online.pth` | `<pack>/*/tracks.npz` |
| 5 solve2 / gate / fusion | `-m laundry_tracking.cotrack.point_solver <shard>`, `-m laundry_tracking.cotrack.gate <shard>`, `-m laundry_tracking.fusion.trajectory_fusion <shard> R,Cg --ct solve3 --suffix _g` | ego | tracks, `out/trackv31speed_<t>.npz`, `out/rig_model_{left,right}.npz`, `body_offset_by_rig_v1.json`, `data/tag_rig_derived/<t>/assignment.json`, color/depth, tagrot, idle_trim | `<pack>/*/solve2_*.npz`, `solve3_*.npz`, `$IC/stitch/qa/fus_{R,Cg}_g/<t>_<arm>_dNN_h*.npz` |
| 6a Cg derived | `-m laundry_tracking.fusion.make_derived <pack>`, env `CGG_SRC=tag_rig_derived CGG_DST=<DC>` | ego | `data/tag_rig_derived/<t>/*`, `stitch/qa/fus_Cg_g`, solve3 | `$L/data/<DC>/<t>/{assignment.json,segments.json,umi_traj_*.npz}` |
| 6b robotize | `$L/datasets/robotize_ik.py $TR/<t> data/<DC>/<t> --demos .. --workers 6` | xvla-mac → `envs/ik` | derived umi_traj, `{wrist,wrist_left}/umi_jaw.npz`, URDF, `configs/robot/*.json`, `visual_teleop/config/{teleop_rebot,calibration}.yaml`, `configs/calibration/umi_gripper_align{,_left}_v001.json` | `data/<DC>/<t>/ik_v1/demoNN.npz`, `robotize_ik.log` |
| 7 quality | `-m laundry_tracking.cotrack.episode_quality q.csv <pack>` + inline pandas | ego (inline used xvla-mac pandas; ego has pandas too) | solve2/3, umi_jaw, `$L/reports/orbbec_v31speed_episode_stats.csv` (fallback for `~/Desktop/orbbec_v31speed_QA_all/episode_stats.csv`, which does not exist) | `q_<PACK>.csv`, `keys_<tag>_gs.txt` |
| build (Mac) | `$L/datasets/tag_rig_unsplit.py --label fused --speed 0.5 --mask-mode stitched --ik-fail-mm 30 --render 0 --grip-label cont` | xvla-mac (lerobot) | raw take (color + wrist jpgs), `umi_jaw.npz`, `data/<DC>/<t>` incl. `ik_v1`, **`reports/idle_trim.csv` (TRIM_CSV)**, split json | `data/lerobot/<name>_train` |

## 2. Local python modules (recursive import closure)

- `$E/scripts`: `jaw_from_motor`, `umi_tcp_traj` → `tag_rig_fuse` → `tag_rig_pose`, `handumi_collector.pose.timing`
- `$E/handumi_collector/{__init__,pose/__init__,pose/estimator,pose/se3,pose/timing}.py`
- repository `src/` (11-04 refactor, packed by `pack_code.sh`, `PYTHONPATH` set by `pod_convert_take.sh`): `laundry_tracking` (the whole pose stage; working files still under `$IC`, the old `$IC` / `$B` / `$F` python files are only used by the CPU CoTracker fallback `run_cotracker_mps.py`), `laundry_common`, `laundry_shared`
- `$L/datasets/{tag_rig_to_lerobot,tag_rig_unsplit,robotize_ik,umi_to_lerobot,xvla_core,wrist_c922}.py`
- `~/laundry_system/contracts/{__init__,action_contract,observation_contract,policy_protocol,skill_contract}.py`
- robotize IK: `$E/visual_teleop/{__init__,config,robot/__init__,robot/ik,robot/ik_pink,tools/__init__,tools/robot_moves}.py`, `$E/ego_teleop/{__init__,robot/*,transforms/{__init__,frames,se3},retarget/{aero_retarget,arm_relative_se3},tracking/interfaces}.py`, `$E/ego_collector/{__init__,hands3d/__init__,hands3d/aero}.py`
- Imports reached only on code paths the chain does not take (left out of the pod envs): `handumi` / `pyroki` (pyroki IK backend; the chain uses pink), `ego_cart20` (`umi_to_lerobot` mcap/visual-jaw paths), `aero_open_sdk`, `requests`, `lerobot` (build only), `torch` (CoTracker only).
- `pack_code.sh` ships **all** `*.py` of these packages and directories, about 900 files and 107 MB including the checkpoint, so lazy imports cannot be missed.

## 3. Static files / weights

- `$E/configs/calibration/*.json` (all <2 MB json/yaml). The chain uses `umi_tag_rig_v001.json`, `umi_tag_rig_v002.json`, `robot_cam_v001..v005.json` (camcal picks one per take date), `jaw_kpt_calib_v002.json`, `umi_gripper_align_v001.json` and `umi_gripper_align_left_v001.json`.
- `$E/configs/robot/{tcp_override,joint_offset,base_override}_v001.json`, which `ik_pink` reads via `__file__`.
- `$E/visual_teleop/config/teleop_rebot.yaml` and `calibration.yaml` (merged).
- URDF: `~/handumi-sw/assets/rebot_b601/rebot_b601_bimanual.urdf`. `ik_pink.URDF = Path.home()/...`; kinematics only, no meshes.
- `laundry_tracking/body_offset_by_rig_v1.json` (package resource), `start_seed_v1.json`, `start_seed_<take>.json` (existing ones are shipped; new ones are made by 3a), `$B/out/rig_model_{left,right}.npz`, `$IC/out/RX_pooled_wrist{,_left}.npy` (not on the Cg path; shipped for safety).
- `$L/reports/idle_trim.csv` (1007 rows; per-job copy) and `$L/reports/orbbec_v31speed_episode_stats.csv`.
- CoTracker3:
  - `~/.cache/torch/hub/facebookresearch_co-tracker_main/{hubconf.py,cotracker/**}` and `~/.cache/torch/hub/checkpoints/scaled_online.pth` (101.7 MB; source `https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth`).
  - With `HOME=<job home>`, `torch.hub` resolves `~/.cache/torch` to the shipped copy. With no internet it falls back to the cached `_main` dir.

## 4. Python environments (versions from `uv pip freeze`, read-only)

Full Mac freezes: `freeze_mac_ego_collector_venv.txt`, `freeze_mac_xvla_mac.txt`. Pod pins: `req_ego.txt`, `req_ik.txt`, `req_ct.txt`.

- **ego** (Mac `~/ego_collector/.venv`, CPython 3.11.15)
  - Pins: numpy 2.4.6, scipy 1.17.1, pandas 3.0.5 (+ pyarrow 25.0.1), pillow 12.3.0, pyyaml 6.0.3.
  - cv2: the venv has both `opencv-python 4.11.0.86` and `opencv-contrib-python 4.14.0.94`. The loaded module reports 4.11.0 and has `cv2.aruco` → the pod pins `opencv-python-headless==4.11.0.86`.
  - Torch is not needed in this env.
- **ik** (Mac `~/xvla-mac`, CPython 3.12.13)
  - Pins: numpy 2.2.6, scipy 1.18.1, opencv-python-headless 4.13.0.92, pyyaml 6.0.3, pandas 3.0.5, `pin` 4.1.0 (pinocchio, cmeel wheels), `pin-pink` 4.4.0, qpsolvers 4.13.0, quadprog 0.1.13 (pink solver = quadprog).
  - qpsolvers optionally imports jax; jax is not required.
  - `-e ~/handumi-sw`, pyroki, lerobot and torch are not needed on the pod.
- **ct** (GPU pod)
  - torch 2.5.0 / torchvision 0.20.0 from the CUDA index (`TORCH_INDEX`, default cu124), numpy 2.4.6, pillow, einops, imageio, matplotlib, tqdm (co-tracker imports).
  - CPython 3.11, matching the Mac env in which `run_cotracker_mps.py` ran.

## 5. Hard-coded paths and how they are overridden

| where | path | override |
|---|---|---|
| everywhere | `Path.home()/ego_collector`, `/laundry_dit_training`, `/laundry_system`, `/handumi-sw`, `~/.cache/torch` | `HOME=$ROOT/jobs/<PACK>/home` (per-job copy of the code template). This is the only mechanism needed: no chain module contains a literal `/Users/...` path (grep 11-02). |
| `tag_rig_to_lerobot.py` | `IK_PY = ~/xvla-mac/bin/python`, `EGO_PY = ~/ego_collector/.venv/bin/python` | symlinks `<home>/xvla-mac → envs/ik` and `<home>/ego_collector/.venv → envs/ego` |
| `fgo_data.py` | `ARCHIVE = /Volumes/PortableSSD/laundry_dit/tag_rig_raw` (meta.json fallback) | harmless (only consulted when `TR/<t>/meta.json` is missing) |
| `episode_quality.py` | `~/Desktop/orbbec_v31speed_QA_all/episode_stats.csv` | absent on the Mac too → the shipped `reports/orbbec_v31speed_episode_stats.csv` alt path is used |
| chain script | scratchpad `S`, `/opt/homebrew/bin/rsync`, Ray `100.64.0.1`, `gpu-4090` | removed: pack and results live in `$ROOT/jobs/<PACK>`, CoTracker is a local call |
| json written by the chain | `assignment.json["take"]` holds the pod path (`/workspace/rp_convert/raw/...`) | informational only; `tag_rig_unsplit` / `robotize_ik` take paths from the CLI. `pod_setup.sh` also links `/Users/jeonghwanlee → code template` as a safety net. |

## 6. Volume layout and per-take I/O

```
/workspace/rp_convert/
  rp_convert_code_<stamp>.tgz      upload_take.sh --code
  code/home/                       template (pod_setup.sh)
  uv-python/  envs/{ego,ik,ct}/    pod_setup.sh
  raw/tag_rig/<take>/              upload_take.sh (4-11 GB: color/depth ~1.2 GB, wrist + wrist_left jpg ~3.2 GB for a 4.4 GB take)
  jobs/<PACK>/home/                per-job HOME (idle_trim.csv, seeds, v2b/out, tagrot, stitch/qa, data/tag_rig_derived*, ...)
  jobs/<PACK>/<PACK>/              CoTracker pack (frames ~25 MB/demo + tracks/solve npz)
  jobs/<PACK>/results/             what fetch_results.sh brings back (no frames)
```

- Steps 0/1 write into `raw/tag_rig/<take>` itself (`umi_jaw.npz`, `umi_traj_*.npz`, `segments.json`), as on the Mac. Do not put one take in two concurrent jobs.
- The Mac build needs, from `results/`: those take files, `data/tag_rig_derived/<t>` and `data/<DC>/<t>` (with `ik_v1`), and the idle_trim rows. `fetch_results.sh INSTALL=1` merges the rows into `reports/idle_trim.csv` (append only, with backup) and installs the rest with `--ignore-existing` (`OVERWRITE=1` keeps backups).

## 7. Run order (no pod launched by these scripts)

1. Mac: `./pack_code.sh` → `dist/rp_convert_code_<stamp>.tgz`; `RP_HOST=root@ip RP_PORT=p ./upload_take.sh --code dist/...tgz`; `./upload_take.sh <take> ...`
   - rsync needs the pod's public-IP SSH port. The `ssh.runpod.io` proxy does not carry rsync/scp.
2. First pod: `tar -xzf /workspace/rp_convert/rp_convert_code_<stamp>.tgz -C /tmp home/laundry_dit_training/remote/runpod_convert/pod_setup.sh && bash /tmp/home/laundry_dit_training/remote/runpod_convert/pod_setup.sh /workspace/rp_convert/rp_convert_code_<stamp>.tgz`. Every other pod: `pod_setup.sh -` (installs rsync/uv in that container, builds missing envs, runs import smoke tests).
3. One pod does everything (GPU pod): `PACK=ctpackFL1102X TAKES="t1 t2" DEMOS="t1=0,1,2" bash .../pod_convert_take.sh`.
   - Split mode instead: `STAGES=pre` on a CPU pod → `STAGES=ct` on the GPU pod → `STAGES=post` on a CPU pod (same PACK, same volume).
   - Several PACKs can run on different pods in parallel.
4. Mac: `RP_HOST=.. RP_PORT=.. INSTALL=1 ./fetch_results.sh ctpackFL1102X`, then the printed `tag_rig_unsplit.py` build.

## 8. Per-pod requirements (estimates, nothing measured)

- **CPU pod (pre/post)**
  - 16–32 vCPU. solve2 is the bottleneck: ~10 min/demo/worker on the Mac M-series. A cloud vCPU (hyperthread) is likely 1.5–2× slower per worker. Shards = min(WORKERS, demos), so a 10-demo take finishes solve2 in roughly 15–20 min with ≥10 vCPU.
  - Prep/robotize IK uses 6 processes per take.
  - RAM 2 GB/vCPU (32–64 GB). Each solve2/fusion/track worker holds one take's arrays plus a few 848×480 depth frames; robotize workers are small. Not measured.
  - Container disk ~10 GB (data is on the volume).
- **GPU pod (ct)**
  - Any CUDA GPU with ≥16 GB (e.g. A4000/A4500/L4/4000 Ada; 4090 24 GB was used before).
  - CoTracker3 online only sends 2×step frames (16 × 848×480) plus ~800–1000 queries per call to the GPU.
  - Host RAM ≥16 GB: one demo video (~200–320 frames) is stacked as float32 on the CPU, ~1.5 GB, plus the uint8 copy.
  - 4+ vCPU (jpg decode). Driver ≥550 for cu124 wheels.
- **Network volume**
  - Per take: the raw take (4–11 GB) + ~0.3 GB of jobs data.
  - Shared: envs ~6–7 GB (ct env CUDA torch ~5 GB, ego ~0.6 GB, ik ~0.4 GB), uv-python ~0.2 GB, code ~0.1 GB.
  - Example: 16 takes × ~6 GB ≈ 100 GB + 10 GB → a 150 GB volume. Pods using it must be in the volume's data center.

## OPEN

1. **Numerical parity** Mac arm64 ↔ Linux x86_64 is unverified (BLAS/LAPACK, `splu`, cv2 build differences). Recommend re-running one already-converted take (e.g. FL1102C `20261102_133624`) on the pod and diffing `q_*.csv` and the GOLD/SILVER keys before trusting new takes.
2. **CoTracker env on the 4090** (`gpu-4090:~/ldt_tagrig/.venv`, the env that produced all previous tracks.npz) was not inspected. torch/CUDA versions there are unknown, so `req_ct.txt` follows the Mac ego venv (torch 2.5.0). The co-tracker hub repo revision on the 4090 is likewise unknown; the Mac cache is from 10-24.
3. **CoTracker on CPU** (`CT_ALLOW_CPU=1`, `run_cotracker_mps.py`) is possible but its speed is unknown. Default is to require a GPU.
4. **pinocchio wheels**: `pin==4.1.0` / `pin-pink==4.4.0` Linux wheels for cp312 are assumed to exist (cmeel). If uv fails, relax to the nearest available versions and re-check IK parity (`ik_v1` npz diff).
5. `opencv-python-headless==4.11.0.86` for the ego env assumes the loaded Mac cv2 is the plain 4.11 wheel. The Mac venv also has opencv-contrib 4.14 installed over the same `cv2` package, so the binary actually loaded could be a mix. Verify tag counts on the parity take (step 1 prints them).
6. Memory/CPU figures in §8 are estimates (nothing was run). Watch the first pod with `free -g` / `top`.
7. `new_take_prep.py` skips takes already present in `idle_trim.csv`. The shipped csv is the Mac copy at pack time, so a re-conversion reuses the Mac rows (same as on the Mac). Rows created on the pod reach the Mac only through `fetch_results.sh INSTALL=1`.
8. `pod_convert_take.sh` and the other shell scripts were only syntax-checked (`bash -n`). `pack_code.sh` was dry-run (`DRY_RUN=1`, 908 files, 107 MB); its tarball was not built. The pod side has not run.
