# 코드 안내: 패키지, 인터페이스, 좌표 표현

*구성 요소 표, 설치 참고, "나눈 방식"은 2026-11-05에 지금 구조에 맞게 다시 썼고, 인터페이스 표와 좌표 표현은 README에서 경로만
고쳐 옮겼습니다. 옛 경로 → 새 경로: [refactor_2026-11.md](refactor_2026-11.md)(영문).*

## 파이프라인 단계별 패키지와 폴더

단계는 [pipeline_ko.md](pipeline_ko.md#실제-데이터-명령-07단계)와 같습니다. 명령줄이 있는 모듈은 모두 `python -m <패키지>.<모듈>`로
실행하고 `main(argv) -> int`가 있습니다. 패키지마다 `src/<패키지>/tests/`에 특성화(characterization) 테스트가 있습니다.

| 패키지 / 폴더 | 단계 | 들어 있는 것 | 상태 |
|---|---|---|---|
| [`src/laundry_common/`](../src/laundry_common/) | 전체 | `paths`: 장비마다 다른 모든 루트(`EGO_DATA_ROOT`, `EGO_FUSION_WORK`, `EGO_TRAIN_ROOT`, `EGO_CKPT_ROOT`, `TAGRIG_STATE_DIR`, …), 환경 변수로 바꿀 수 있고 기본값은 녹화 장비의 경로 | 현재 |
| [`src/laundry_tracking/`](../src/laundry_tracking/) | 0, 2–4 | `calib/`(카메라 ↔ 로봇 `robot_cam_solve` / `robot_cam_from_anchors`, 태그 리그 `tag_rig_solve` / `tag_rig_pivot` / `tag_rig_touch_tcp`, 태그 → 집게 끝 궤적 `umi_tcp_traj`), `new_take_prep`, `cotrack/`(`make_pack`, `run_cotracker`, `point_solver`, `gate`, `episode_quality`), `fusion/`(`trajectory_fusion` 변형 Cg, `make_derived`, `render_qa`, `render_sources`), trim `tighten_trim`, `edge_gap_trim`, `fgo/`(데이터 로더, 포즈 smoother) | 현재 (CoTracker는 CUDA GPU 필요) |
| [`src/laundry_jaw/`](../src/laundry_jaw/) | 2 | 집게 상태: `from_motor`(서보 → `umi_jaw.npz`, 10-29부터의 출처), `aperture` / `kpt_label` / `kpt_train`(손목 영상의 집게 끝 keypoint), `motorlike`(+ `motorlike_overlay`, 재라벨 시도, 보고한 모델에는 쓰지 않음) | 현재 |
| [`src/laundry_dataset/`](../src/laundry_dataset/) | 5 | 라벨 + LeRobot 내보내기: `robotize_ik`(라벨 F / A / B + IK 진단, IK는 하위 프로세스 `EGO_IK_PYTHON`), `tag_rig_unsplit`(데이터셋 빌더), `tag_rig_to_lerobot`, `idle_trim`, `ik_qa_stats`, `builds/`(11-02 → 11-05 펴기 세트의 split / 점검 / 업로드 / QA 영상 도구) | 현재 |
| [`src/laundry_training/`](../src/laundry_training/) | 6 | LeRobot 포크 위의 DiT / ACT 학습기: `env_options`(모든 스위치: `GRIP_STATE_MASK`, `GRIP_BCE`, `GRAD_CLIP`, `MASK_CROSS_GAP`, `AUG_*`, `PGRASP`), `datasets`, `loss_masks`, `augment`, `grip_head`, `train_dit_tagrig`, `train_dit_mix` + 설정 | 현재 (학습 노드의 LeRobot 포크 필요) |
| [`src/laundry_deploy/`](../src/laundry_deploy/) | 7 | receding-horizon deploy: `core/`(정책 / X-VLA / 라벨 runner, chunk, real-time chunking, 운동학, 충돌, 높이 캘리브레이션), `session/`(실행 루프, streamer, 카메라, CLI), `ui/`(Flask UI :8770), `tools/`(오프라인 평가, 높이 캘리브레이션, 카메라 정렬 / 보기, …), `robot_service/`(reBot MIT 브리지 + 집게 / 관절 영점 도구), `xvla_server` | 현재 (움직이는 것은 로봇 + reBot 제조사 플러그인 필요) |
| [`src/laundry_reference/`](../src/laundry_reference/README.md) | 3, 5, 9 | 궤적 단계의 작은 독립 재구현(궤적 CSV, 끊김 보완, relative SE(3), 로봇화, 비교 지표, 그림), 합성 데이터로 돌아감. 실제 파이프라인은 쓰지 않음 | 현재 (참고 구현) |
| [`src/laundry_shared/`](../src/laundry_shared/) | 1, 2, 재생, 7 | 빨래 코드가 HandUMI / teleop 작업에서 import하는 모듈의 사본, 원래 스타일 그대로: `visual_teleop`(녹화기 `tag_rig_record`, 집게 분류기 `umi_jaw`, reBot IK·제한, 재생 계획 `umi_replay_bi`), `rebot_prep.parity`(태그 리그 위치), `handumi_collector`, `ego_collector`, `ego_teleop`, `contracts`, `xvla_core`, `umi_to_lerobot` ([출처](provenance.md)) | 현재 (실시간 teleop은 보류) |
| [`scripts/laundry/`](../scripts/laundry/) | 3, 6, 재생 | 운영용 실행 위치: `eval/`(재생 비교, 나란히 보기 영상, 계획 회귀, 클러스터 학습의 에포크별 오프라인 평가), `retarget/`(UMI ↔ reBot EEF 정렬, 양손 로봇화), `ray_cotracker/`와 `ray_ditddp/`(클러스터 작업), `runpod_convert/`(임대 pod에서 변환 전체), `grasp/`(11-05 잡음 / 빈손 라벨과 p_grasp 부가 파일, 연구용 script, 아직 OA 스타일 아님) | 현재 |
| [`configs/laundry/`](../configs/laundry/README.md) | 0 | 캘리브레이션 결과, 로봇 설정과 보정값, 집게 모델, 회귀 기준값, HandUMI 프로필, 고정 버전 환경(`envs/`) | 현재 |
| [`tests/laundry/`](../tests/laundry/) | – | 구조 / import smoke test, script의 합성 자체 시험, deploy UI 경로, 참고 구현 회귀(`expected_synthetic.json`) | 현재 |
| [`assets/laundry/examples/`](../assets/laundry/examples/README.md) | – | 합성(SYNTHETIC) 양손 시연 + 시뮬레이션 재생(CSV / NPZ)과 생성기 | 현재 |
| [`hardware/umihand/`](../hardware/umihand/README.md) | 1 | HandUMI 제작 사본 + 우리 변경 부품, AprilTag 시트, IMU 펌웨어 (원 설계: BrikHMP18, Apache-2.0) | 현재 |
| [`analysis/`](../analysis/verify_claims.py) | – | `verify_claims.py`: 테크니컬 리포트의 모든 숫자를 그것을 옮겨 온 문서와 대조 | 현재 |
| [`report/`](../report/README.md) | – | 테크니컬 리포트, 영어 + 한국어 (PDF + LaTeX + 그림), 이전 버전은 `report/archive/` | 현재 |
| [`docs/`](../docs/) | – | 이 안내, [파이프라인](pipeline_ko.md), [이력](history_ko.md), [세부 내용](details_ko.md), [결과](results.md)(영어), [재생 트랙 단계 표](pipeline_replay_track.md), [환경](environments.md), [출처](provenance.md), 날짜별 상태 노트, [기록](records/README.md) | 현재 |
| [`media/`](../media/README.md) | – | 실제 장치의 GIF, 영상, 사진 | 현재 |
| [`archive/`](../archive/README.md) | – | MASt3R 끊김 보완과 러너(10-27 이동), ORB-SLAM3, UMIalign, 스테레오 VIO, 빨간 테이프 집게, 맨손 수집기, 옛 구조를 설명하던 문서 | ~~더 이상 안 씀~~ |

## 설치 참고

```bash
uv sync                               # Python 3.12, 기본 의존성(numpy, scipy, pandas, pyyaml, matplotlib, OpenCV < 5) + pytest
uv sync --group vision                # + torch, torchvision, ultralytics (CoTracker 단계, 집게 모델)
uv sync --group robot                 # + pin, pin-pink, requests, flask (IK 라벨, deploy, 로봇 브리지)
uv sync --group train                 # + torch, diffusers, accelerate, safetensors (학습기는 LeRobot 포크도 필요)
uv sync --group collector             # + pyserial, feetech-servo-sdk, av, mcap, tqdm, pupil-apriltags (녹화기)
# uv 없이:
python -m pip install -e ".[test]"    # 기본 의존성 + pytest
python -m pytest -q
```

학습·배포·로봇 브리지 환경(결과를 만든 장비에서 기록한 고정 버전, 이 레포만으로는 재현할 수 없는 부분)은
[environments.md](environments.md)에 있습니다.

설치하지 않을 때는 `src/`를 경로에 넣습니다: `PYTHONPATH=src python -m laundry_<단계>.<모듈>`. `scripts/laundry/`의 script는
패키지를 설치하거나 `PYTHONPATH=src`를 둔 상태에서 파일로 실행합니다(`python scripts/laundry/eval/replay_parity.py …`).

참고:
- 원래 실행은 여러 인터프리터로 했습니다: 녹화기는 저장소 `.venv`(macOS에서 Orbbec 때문에 `sudo`), IK·브리지 시작·재생은
  Pink / pinocchio가 있는 X-VLA 환경, 집게 keypoint script는 별도 ultralytics 환경. 코드가 프로젝트의 다른 작업 폴더에서
  import하던 보조 모듈은 `src/laundry_shared/`에 복사해 두었습니다([출처 + md5](provenance.md)). 아직 밖에 있는 것: reBot 제조사
  LeRobot 플러그인, X-VLA 학습 패치.
- 패키지는 기본값으로 원래 장비의 데이터 폴더를 읽고 씁니다. 다른 위치를 쓰려면 `EGO_DATA_ROOT`(녹화, 캘리브레이션),
  `EGO_FUSION_WORK`(fusion 작업 폴더), `EGO_TRAIN_ROOT`(파생 궤적, 리포트, 데이터셋), `EGO_CKPT_ROOT`(checkpoint 보관)를 설정합니다
  ([`src/laundry_common/paths.py`](../src/laundry_common/paths.py)). 학습과 X-VLA 서버는 기본 설치가 아니라 LeRobot / X-VLA
  환경(torch, lerobot, diffusers)이 필요합니다.
- 포함하지 않은 것: 원본 데이터, 학습된 checkpoint, reBot 제조사 플러그인과 URDF, ORB-SLAM3 소스.

## 나눈 방식

**세 층.** 어느 코드가 무엇인지 한눈에:

| 층 | 위치 | 무엇인가 |
|---|---|---|
| 실제 코드 | `src/laundry_{common,tracking,jaw,dataset,training,deploy}/`, `src/laundry_shared/`, `scripts/laundry/` | 보고한 모든 결과를 만들고 로봇을 돌리는 코드(실제로 돌린 코드와 동작이 같은지는 특성화 테스트로 확인) |
| 이식용 참고 구현 | [`src/laundry_reference/`](../src/laundry_reference/README.md) | 핵심 아이디어를 실험실 장비 없이 합성 데이터로 돌릴 수 있게 다시 짠 작은 구현. 실제 파이프라인은 쓰지 않음 |
| 대체됨 | [`archive/`](../archive/README.md) | 기록으로 남긴 이전 방식 ([이력](history_ko.md#지금-쓰는-것과-안-쓰는-것) 참고) |

* **`src/`**에는 import하는 코드가 모두 있고, 파이프라인 단계마다 패키지 하나입니다(위 표). `laundry_shared`는 깨끗한 clone에서도
  import가 풀리도록 HandUMI / teleop 작업에서 복사해 온 모듈이고 원래 스타일을 유지합니다.
* **`scripts/laundry/`**에는 얇은 운영용 실행 위치가 있습니다: 클러스터·pod 작업과 재생 트랙 script.
* **`configs/laundry/`**에는 코드가 `<저장소 루트>/configs/laundry/...`로 읽는 캘리브레이션·로봇 파일이 있습니다
  (`laundry_common.paths.CONFIGS` 또는 `Path(__file__).parents[n]`).
* **`assets/laundry/examples/`** — 합성 데이터와 결정적 생성기. **`tests/laundry/`** — import·경로 smoke test, script의 합성 자체
  시험, 참고 구현 회귀 검사.
* **`docs/`, `report/`, `media/`** — 세부 내용, 결과, 테크니컬 리포트, 그림과 실제 장치 영상. **`hardware/`** — HandUMI 제작 사본.
  **`archive/`** — 더 이상 안 쓰는 코드.

## 실제 파이프라인의 인터페이스

정의한 코드 기준:

| 인터페이스 | 정의 | 위치 |
|---|---|---|
| 월드 좌표 | reBot 베이스 좌표 B. 카메라 → 베이스 `T_B_C`는 `configs/laundry/calibration/robot_cam_v00N.json` (take 날짜로 고름) | `src/laundry_tracking/camera_calibration.py`, `src/laundry_tracking/calib/umi_tcp_traj.py` |
| take별 집게 끝 궤적 | `umi_traj_{right,left}.npz`: 집게 끝 TCP `p_B`, 리그 회전 `R_B_tcp`, `ntags` | `src/laundry_tracking/calib/umi_tcp_traj.py` |
| 집게 상태 | `umi_jaw.npz`: `g_cont` 0 = 닫힘 … 1 = 열림, `valid`, `ev_t`. 이진 `state`는 1 = 열림, 0 = 닫힘. 서보에서 만든 파일은 `motor_pos`를 함께 두고 `model = feetech_sts3215_motor` | 영상: `src/laundry_shared/visual_teleop/tools/umi_jaw.py`, `src/laundry_jaw/aperture.py`, 서보(Feetech STS3215, 토크 끈 엔코더, 10-29 →): `src/laundry_jaw/from_motor.py` |
| 학습 라벨 | **F**(10-29 →, `--label fused`) = reBot 집게 축(`R_B_tcp·R_U_E`)으로 나타낸 fusion 사람 집게 끝 포즈, 0.25초 smoothing, IK 이전. 테이블 들어올림·clamp 없음, IK는 값도 바꾸지 않고 mask도 하지 않음. ~~**A**(`--label rob`) = F + 테이블 여유(수직 쪽으로 ≤ 30° 기울인 뒤 corner lift)~~. **B**(`--label ikfk`) = IK → FK (QA용) | `src/laundry_dataset/robotize_ik.py` (`laundry_dataset.tag_rig_unsplit`의 `--label rob` / `ikfk` / `fused`) |
| 데이터셋 | LeRobot v3, 20 Hz, 카메라 3대. `observation.state` 30차원 = 팔마다 에피소드 시작 기준 상대 포즈(pose9), g_L, g_R, 좌우 상대 pose9 + 유효 플래그(`GRIP_STATE_MASK=1`이면 집게 값은 0). `action` 20차원 absolute `[L xyz, L rot6d, R xyz, R rot6d, g_L, g_R]`(rot6d = R의 앞 두 행, 집게 목표 = 0.1초 ramp를 둔 이진 집게 상태, 연속 열림 정도는 `gripper_cont`로 따로 저장), 차원별 `action_mask[20]` | `src/laundry_dataset/tag_rig_unsplit.py`, `src/laundry_dataset/tag_rig_to_lerobot.py`, `src/laundry_shared/contracts/` |
| chunk | relative 동작 `inv(T_t)·T_{t+k}`, k = 1…32, 20 Hz, 학습 loader에서 만듦. deploy는 계획마다 앞의 몇 step만 실행하고 다시 계획(DiT 기본 4, ACT 1, X-VLA 8, UI에서 선택, 속도 > 1이면 DiT 실행 step은 32/속도로 제한) | [details_ko.md 3.2–3.3절](details_ko.md#33-학습한-정책을-로봇에서-실행하기) |
| 로봇 모델 보정 | TCP 오프셋, 팔별 관절 영점 오프셋, 왼팔 베이스 자세: `configs/laundry/robot/{tcp_override,joint_offset,base_override}_v001.json` | `src/laundry_shared/visual_teleop/robot/ik_pink.py`에서 적용 |
| 단위 | 코드는 미터와 초로 저장(예: 태그 크기 0.048 m). README와 리포트는 mm로 적음 | — |

## 참고 구현 인터페이스: 궤적 CSV

`laundry_reference` 도구는 모두 넓은 CSV 하나를 주고받습니다. 정의는 [`src/laundry_reference/trajio.py`](../src/laundry_reference/trajio.py), 예제
데이터와 함께 [`assets/laundry/examples/README.md`](../assets/laundry/examples/README.md)에 설명이 있습니다. 프레임마다 한 행, `t`는 초, 이어서 손마다(`R_` 오른쪽,
`L_` 왼쪽):

| 열 | 단위 | 뜻 |
|---|---|---|
| `x, y, z` | m | 월드(로봇 베이스) 좌표의 집게 끝 위치: +x 앞, +y 왼쪽, +z 위, 테이블 평면 z = 0 |
| `qx, qy, qz, qw` | – | 집게 끝 방향, 단위 quaternion |
| `jaw` | – | 1 = 열림, 0 = 닫힘 |
| `visible` | – | 1 = 포즈 측정됨(마커 보임), 0 = 측정 안 됨(포즈는 NaN) |

도구가 덧붙이고 이후 도구가 그대로 넘기는 열:

| 열 | 쓰는 곳 | 뜻 |
|---|---|---|
| `filled` | `gap_fill` | 0 측정 / 1 보간 / 2 보완 안 함(너무 길거나 끝이 열림) |
| `human_x/y/z` | `robotize` | 로봇화 이전 궤적 |
| `disp_mm` | `robotize` | 프레임별 로봇화 변위, mm |
| `quality` | `robotize` | 0 / 1 / 2 (기본 경계 20 / 50 mm) |

## 좌표 표현

* **저장: 월드 좌표의 absolute 포즈.** 프레임마다 손별 로봇 베이스 좌표 `B`(+x 앞, +y 왼쪽, +z 위)의 집게 끝 포즈
  `T_B(t) ∈ SE(3)`, 집게 상태, 포즈를 측정했는지 보완했는지 표시를 저장합니다. 집게 끝 TCP는 사람 집게와 로봇 집게에서 같은
  방식(두 집게 끝의 중점)으로 정의하므로 나중에 공구 오프셋을 추측할 필요가 없습니다.
* **학습 계약: relative SE(3).** 학습 목표는 기준 포즈 `T_0`(예: action chunk 시작 포즈)에 대한 상대값으로 표현합니다.

  `T_rel(t) = T_0^{-1} T(t)`

  그래서 목표가 작업 공간 어디에서 움직였는지에 의존하지 않고, 같은 계약이 사람 데이터와 로봇 데이터에 똑같이 적용됩니다.
  absolute로 저장해 두면 어떤 relative 형태든 다시 만들 수 있고, 재생도 absolute를 씁니다. 변환과 왕복 검사의 기준 구현:
  [`src/laundry_reference/relative_se3.py`](../src/laundry_reference/relative_se3.py) (chunk 길이 `--chunk`, 기본 1.0초).
* **프레임별 품질.** 로봇화는 프레임마다 로봇 목표를 사람 궤적에서 얼마나 옮겼는지와 거기서 나온 대략적 품질 등급을 저장하고,
  보완한 프레임은 따로 표시합니다. 라벨 가중치로 쓰려는 것입니다.
