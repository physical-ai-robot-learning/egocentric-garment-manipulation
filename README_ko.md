# 양손 옷 다루기를 위한 egocentric 사람 시연

손에 끼는 평행 집게(HandUMI)로 녹화한 사람의 티셔츠 시연을 양팔 로봇(**reBot B601-DM**)의 베이스 좌표에서 로봇이 할 수 있는
집게 끝 궤적과 학습 라벨로 변환합니다. 먼저 데이터 전체를 끝까지 시험하기 위해 open-loop로 재생하고, 그다음 티셔츠 펴기·접기
정책 학습에 씁니다. 로봇 데이터 수집이 로봇 수가 아니라 사람 수에 따라 늘어나게 하려는 것입니다.

**[physical-ai-robot-learning](https://github.com/physical-ai-robot-learning)**의 Physical AI / 로봇 학습 프로젝트 중 하나입니다.

[![ci](https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange.svg)](LICENSE)
· **[테크니컬 리포트 (PDF, 영문 11-12)](report/report_en.pdf)** ([한국어 11-02](report/report_ko.pdf)) · [이전 버전](report/archive/)
· [자세한 결과](docs/results.md) · [프로젝트 이력](docs/history_ko.md) · [파이프라인](docs/pipeline_ko.md)
· [English README](README.md) · [출처](ATTRIBUTION.md) · 기준일: 2026-11-12

![사람 vs 로봇 재생](media/human_vs_robot_replay_large.gif)

*사람 시연(왼쪽) vs 변환한 궤적의 로봇 재생(오른쪽). 같은 고정 카메라, 하나의 시계로 맞췄고 사람 속도로 보여 줍니다. 막대는
로봇 집게 명령입니다.*

## 현재 상태 (2026년 11월 12일)

> **되는 것과 측정 방법**
> * **재생 (10-08 → 10-22).** 33초 데모 하나의 양팔 재생은 결정적이고(실행 사이 약 1 mm), 집게가 사람과 같은 높이·시점에
>   닫힙니다. 천을 집어 끌지만 재생으로 접기는 **성공하지 못합니다**. 고정 카메라, 줄자, 로봇 엔코더로 측정했습니다([결과](docs/results.md)).
> * **학습 데이터 (10-26/27).** 펴기 성공 시연 300개를 등급을 매긴 IK 이전 Cartesian 학습 라벨로 바꿨습니다. 위치는 손목 SLAM이
>   아니라 마커를 기준점으로 쓰는 고정 카메라 점 추적에서 얻습니다.
> * **로봇에서 돌린 첫 정책 (10-28 → 11-01).** GOLD 데모 281개로 파인튜닝한 DiT·X-VLA가 천 쪽으로 가지만 닿기 전에 멈추고 집게를
>   일찍 닫습니다. 기록된 원인과 그 뒤 데이터 수정:
>   [이력](docs/history_ko.md#로봇에서-돌린-첫-정책과-데이터에서-드러난-것-10-28--11-01).
> * **핀칭 (11월 2일).** 범위를 줄인 시험(왼손으로 소매 하나 집기)에서 핀칭 시연 197개로 파인튜닝한 DiT(50 에포크 checkpoint)가
>   소매를 집어 당겼습니다. 11월 2일 실제 로봇 시행에서 집기 성공을 관찰했고, **시행 횟수는 세지 않았습니다**. 놓친 뒤에는 다시
>   시도하지 않고 다른 곳을 잡습니다(데이터에 복구 시연이 없음).
> * **핀칭 뒤의 펴기 (11-02 → 11-05).** FLAT149(UMI 천 접기 레시피), 그리고 11월 3일 방향을 바꾼 카메라 아래의 FLAT250(train
>   시연 250개, 시연 안의 추적 gap으로 split): 13 에포크까지 held-out 점수 72 → 52 mm. 집게 서보에는 부하 신호가 없어서, 닫힌
>   집게가 천을 잡았는지를 손목 카메라 분류기가 프레임마다 추정합니다(p_grasp, 11월 3–5일). 이 정책들의 시행 횟수를 센 로봇 시험은
>   아직 없습니다.
> * **p_grasp를 넣은 펴기, 로봇에서 (11-05 → 11-07).** FLAT-PGRASP: FLAT250 레시피를 11월 3–5일 데이터(train 301 / val 46 /
>   test 45 시연)에 쓰고, 상태의 집게 칸에 p_grasp를 넣었습니다. 0.3배로 늘린 데이터에서 held-out 점수는 46.04 / 46.61 mm(val /
>   test, 53 에포크). 로봇에서는(11월 6일) 0.3배 정책이 같은 레시피의 1.0배보다 나아 보였고, 11월 7일에는 두 집게가 셔츠
>   가장자리를 잡아 두 팔 사이로 폈습니다([아래 중간 보고](#최근-로봇-실행-중간-보고-11월-7일)). 셔츠를 평평하게 내려놓는 데까지는
>   아직 못 갑니다. 로봇에서 p_grasp는 손목 카메라가 아니라 집게 상태에서 얻습니다. 사람 데이터로 학습한 분류기가 로봇의 닫힌 빈
>   집게를 '잡음'으로 보기 때문입니다. 로봇 시행 횟수는 세지 않았습니다.
> * **로봇의 지연 (11-06).** 로봇 집게가 늦게 닫혔습니다. 명령→동작 지연을 재 보니 팔 58–64 ms, 집게 171–173 ms였고, UMI처럼
>   이 지연만큼 명령을 앞당겨 보내고 상태는 영상을 찍은 시각 기준으로 잡습니다.
> * **두 번째 정책 계열: 같은 데이터로 X-VLA (11-10 →).** 우리 SoftFold X-VLA 체크포인트(650k)에서 새 domain id로 파인튜닝: RTX 4090 2장에서
>   5 에포크, 이어서(11-12, 학습 중) H100 4장에서 시연 330개(위 301개 + tag-free 29개)로 100 에포크까지. 오프라인·로봇 결과는
>   아직 없습니다.
> * **마커 없는 수집 (11-11 →).** 정책은 상대 동작으로 움직이므로 라벨에 필요한 것은 마커의 절대 위치가 아니라 시작 자세와
>   방향입니다. AprilTag를 모두 뗐습니다(손 장치, 테이블 시작 태그). 시연은 검은 L 테이프에서 시작하고, 위치
>   체인은 시작 시드를 depth에서, 방향을 정지 때의 IMU에서 얻습니다. 태그가 있는 take를 태그 없이 처리해 태그와 비교하면 집게 끝
>   오차가 왼쪽 17 mm, 오른쪽 9–14 mm입니다. 첫 세트: 11-11 train 29개, 11-12 지금까지 16개 더.
>
> **보여 주지 않는 것:** 성공률 없음(재생이든 어떤 정책이든 로봇 시행 횟수를 세지 않았음), 장치 하나·시연자 한 명, 펴기는 아직
> 끝까지 못 하고 접기는 못 함. 11-06 → 11-12 자세한 내용: [11-12 상태 노트](docs/status_2026-11-12.md)(영문).

### 최근 로봇 실행 (중간 보고, 11월 7일)

![로봇에서 돌린 펴기 정책, run 165051](media/flatten_policy_rollout_20261107_165051.gif)

*0.3배 FLAT-PGRASP(77 에포크 checkpoint)를 로봇에서 돌린 run 165051, 고정 카메라, 처음 97초, 제어 주기(약 1.3초)마다 한
프레임, 약 6배속. 구겨진 셔츠에서 시작해 약 25초 안에 두 집게가 가장자리를 하나씩 잡고 두 팔 사이로 셔츠를 펴서 그 상태를
유지합니다. 더 지나면 셔츠가 다시 뭉칩니다. 결과가 아니라 진행 상황을 알리는 중간 보고입니다. 이날 돌린 27번 중 가장 잘 된 것을
골랐고, **시행 횟수를 세거나 채점하지 않았으므로** 정책이 할 수 있는 동작을 보여 줄 뿐, 얼마나 자주 되는지는 보여 주지 않습니다.
자세한 내용: [상태 노트](docs/status_2026-11-07.md)(영문).*

### 태그 없는 변환 (11월 12일)

![태그 없는 집게 끝 추적, take 20261112_135910 demo 2](media/tagfree_20261112_135910_d02.gif)

*AprilTag를 모두 뗀 뒤 녹화한 펴기 시연(take 20261112_135910, demo 2, GOLD, train)을 태그 없는 체인으로 변환한 결과. 왼쪽:
고정 카메라 위에 집게 끝 자세(파랑 = 왼쪽, 주황 = 오른쪽; 빨강 = x, 초록 = z 축, 5 cm)와 최근 궤적. 오른쪽: 손목 카메라 두 대.
시작 자세는 검은 L 테이프와 정지한 장치의 depth에서, 방향은 정지 때의 IMU에서 얻습니다. 약 2배속. [MP4](media/tagfree_20261112_135910_d02.mp4) ·
[상태 노트](docs/status_2026-11-12.md) §4(영문).*

**연구 질문.** *로봇이 변환된 사람 동작을 open-loop로 그대로 재생하면, 사람이 한 일을 하는가?* 재생은 모든 단계(캘리브레이션,
위치, 집게 상태, 로봇화, IK, 시간, 제어)를 한꺼번에 거치고, 사람·로봇 명령·측정한 로봇 동작을 하나의 시계와 하나의 좌표에 두어
오차가 어느 단계에서 생겼는지 가를 수 있게 하며, 실패가 눈에 보입니다. 같은 변환 데이터가 그다음 정책 학습 라벨이 됩니다.
동기와 부수 질문: [이력](docs/history_ko.md#동기와-연구-질문).

## 저장소 지도

| 경로 | 무엇 | 이럴 때 읽기 |
|---|---|---|
| [`src/laundry_tracking/`](src/laundry_tracking/) | 0, 2–4단계: 카메라 ↔ 로봇과 태그 리그 캘리브레이션(`calib/`), CoTracker + 마커 + IMU fusion(`cotrack/`, `fusion/`), trim, GOLD / SILVER / REJECT 등급 | 로봇 좌표에서 집게 끝 포즈를 어떻게 복원하는지 보고 싶을 때 |
| [`src/laundry_jaw/`](src/laundry_jaw/) | 2단계: 서보(`from_motor`)나 손목 영상의 집게 끝 keypoint(`aperture`)로 집게 상태 | 집게 라벨이 어디서 오는지 알고 싶을 때 |
| [`src/laundry_grasp/`](src/laundry_grasp/README.md) | 5b단계: 손목 카메라로 집게 닫힘 이벤트의 잡음 / 빈손 라벨, 고정된 p_grasp 생성기, 프레임별 사이드 파일 | 닫힌 집게가 천을 잡았는지 정책에 어떻게 알려 주는지 보고 싶을 때 |
| [`src/laundry_dataset/`](src/laundry_dataset/) | 5단계: 라벨 F / A / B와 IK 진단(`robotize_ik`), LeRobot 데이터셋 빌더(`tag_rig_unsplit`), split / 점검 빌드(`builds/`) | 라벨 정의와 데이터셋 계약을 보고 싶을 때 |
| [`src/laundry_training/`](src/laundry_training/) | 6단계: DiT 학습기, loss mask, augmentation, 집게 head, 스위치(`env_options`) | 학습 레시피를 보고 싶을 때 |
| [`src/laundry_deploy/`](src/laundry_deploy/) | 7단계: receding-horizon deploy(`core/`, `session/`, `ui/`), 오프라인 평가와 높이 캘리브레이션(`tools/`), reBot MIT 브리지(`robot_service/`) | 정책이 로봇에서 어떻게 돌고 움직이기 전에 무엇을 검사하는지 알고 싶을 때 |
| [`src/laundry_shared/`](src/laundry_shared/) | 1단계와 재생: 녹화기 `tag_rig_record`, 집게 분류기, reBot IK, 재생 계획(파이프라인이 import하는 HandUMI / teleop 모듈의 사본) | 녹화기나 재생 계획을 보고 싶을 때 |
| [`src/laundry_reference/`](src/laundry_reference/README.md) | 궤적 단계의 작은 독립 재구현, 합성 데이터로 실행 가능 | 실험실 장비 없이 핵심 아이디어를 보고 싶을 때 |
| [`src/laundry_common/`](src/laundry_common/paths.py) | `paths`: 장비마다 다른 모든 루트, 환경 변수로 바꿀 수 있음 | 다른 폴더에서 코드를 돌리고 싶을 때 |
| [`scripts/laundry/`](scripts/laundry/) | 재생 비교·리타깃 script, 클러스터 / pod 작업(CoTracker, DiT DDP, X-VLA, 변환), X-VLA 데이터 변환기, 태그 없는 위치 체인(11-12 작업 코드 스냅샷, [`tagfree/`](scripts/laundry/tagfree/README.md)) | 운영용 실행 위치를 찾을 때 |
| [`configs/laundry/`](configs/laundry/README.md) | 캘리브레이션 결과, 로봇 보정값, 집게 모델, 고정 버전 환경 | 코드가 읽는 숫자를 보고 싶을 때 |
| [`tests/laundry/`](tests/laundry/), `src/laundry_*/tests/` | smoke test와 패키지별 특성화 테스트 | 코드가 도는지 확인하고 싶을 때 |
| [`assets/laundry/examples/`](assets/laundry/examples/README.md) | 합성(SYNTHETIC) 시연 + 시뮬레이션 재생 | 참고 구현을 돌릴 데이터가 필요할 때 |
| [`analysis/verify_claims.py`](analysis/verify_claims.py) | 리포트의 모든 숫자를 그것을 옮겨 온 문서와 대조 | 리포트를 확인하고 싶을 때 |
| [`docs/`](docs/) | [파이프라인](docs/pipeline_ko.md), [코드 안내](docs/codebase_ko.md), [이력](docs/history_ko.md), [세부 내용](docs/details_ko.md), [결과](docs/results.md)(영어), [환경](docs/environments.md), 상태 노트 [11-05](docs/status_2026-11-05.md) / [11-07](docs/status_2026-11-07.md) / [11-12](docs/status_2026-11-12.md), [경로 대응표](docs/refactor_2026-11.md) | 긴 설명을 원할 때 |
| [`report/`](report/README.md) | 테크니컬 리포트(영어 11-12, 한국어 11-02), LaTeX + PDF | 방법과 결과 전체를 한 문서로 보고 싶을 때 |
| [`media/`](media/README.md) | 실제 장치의 GIF, 영상, 사진 | — |
| [`hardware/umihand/`](hardware/umihand/README.md) | HandUMI 제작 사본(BrikHMP18의 설계) + 우리 변경 부품, IMU 펌웨어 | 집게를 만들고 싶을 때 |
| [`archive/`](archive/README.md) | 더 이상 안 쓰는 코드: MASt3R / ORB-SLAM3 위치, UMIalign, 스테레오 VIO, 빨간 테이프 집게, 맨손 수집기 | 무엇을 시도하고 버렸는지 보고 싶을 때 |

**여기서부터 읽기:** [`docs/pipeline_ko.md`](docs/pipeline_ko.md)(월드 좌표와 단계),
[`src/laundry_tracking/fusion/trajectory_fusion.py`](src/laundry_tracking/fusion/trajectory_fusion.py)(위치 소스),
[`src/laundry_dataset/tag_rig_unsplit.py`](src/laundry_dataset/tag_rig_unsplit.py)(라벨 → 데이터셋).

## 파이프라인

모든 것을 하나의 호스트 시계로, 로봇 베이스 좌표에서 표현합니다. 고정 카메라를 로봇에 캘리브레이션하고, 매 테이크는 로봇 좌표를
한 번 측정해 둔 시작점 마커 두 장(ID 400 / 401)에 집게를 두고 시작합니다. 패키지를 설치한 상태(또는 `PYTHONPATH=src`)로 저장소
루트에서 실행하며, 이 저장소에 없는 녹화 take가 필요합니다. 옵션, 출력, 재생 트랙: [docs/pipeline_ko.md](docs/pipeline_ko.md).

| 단계 | 모듈 | 명령 |
|---|---|---|
| 0 · 캘리브레이션 (한 번, 카메라 ↔ 로봇은 카메라를 옮길 때마다) | `laundry_tracking.calib` | `python -m laundry_tracking.calib.robot_cam_from_anchors <take> --frames 2400:2828:4` (첫 풀이: `calib.robot_cam_solve`, 리그: `calib.tag_rig_solve`) |
| 1 · 녹화 | `laundry_shared.visual_teleop.tools.tag_rig_record` | `sudo .venv/bin/python -m laundry_shared.visual_teleop.tools.tag_rig_record --proto task --wrist` |
| 2 · 태그 위치 + 집게 상태 | `laundry_tracking.calib.umi_tcp_traj`, `laundry_jaw.from_motor` | `python -m laundry_tracking.calib.umi_tcp_traj <take> --hand <hand> --depth --imu-fill`, `python -m laundry_jaw.from_motor <take>` |
| 3 · 점 추적 + 마커 + IMU fusion | `laundry_tracking` | `python -m laundry_tracking.new_take_prep <take>` → `cotrack.make_pack` → `cotrack.run_cotracker`(GPU) → `cotrack.point_solver` → `cotrack.gate` → `python -m laundry_tracking.fusion.trajectory_fusion <pack> Cg --ct solve3` |
| 4 · trim + 품질 등급 | `laundry_tracking` | `python -m laundry_tracking.tighten_trim <idle_audit.csv> 0.3`, `edge_gap_trim <pack>`, `cotrack.episode_quality <quality.csv> <pack>` |
| 5 · 라벨 + LeRobot 데이터셋 | `laundry_dataset` | `python -m laundry_dataset.robotize_ik <take> <derived take> --workers 8`, `python -m laundry_dataset.tag_rig_unsplit --label fused --speed 0.3 --out <dataset> --takes <take> …` |
| 5b · grasp 라벨(p_grasp) | `laundry_grasp` | `python -m laundry_grasp.generate`(고정 `grasp-v2-W0.4`) → `python -m laundry_grasp.sidefile <set> <pgrasp dir>`. 라벨링 페이지 `python -m laundry_grasp.gold_ui <round dir>` |
| 6 · 학습 | `laundry_training` | `python -m laundry_training.train_dit_tagrig …` (LeRobot 포크, 클러스터 작업 `scripts/laundry/ray_ditddp/run.sh`) |
| 7 · deploy + 평가 | `laundry_deploy` | `python -m laundry_deploy.ui.server` (UI :8770, DRY 먼저), `python -m laundry_deploy.session.cli run --policy dit --secs 30`, `python -m laundry_deploy.tools.offline_eval --ckpt … --data … --out …` |
| 재생 트랙 (10-08 → 10-22) | `scripts/laundry/retarget`, `umi_replay_bi`, `scripts/laundry/eval` | `python scripts/laundry/retarget/umi_robotize_bi.py <take> …` → `python -m laundry_shared.visual_teleop.tools.umi_replay_bi <take> … --offline` → `python scripts/laundry/eval/replay_parity.py <take> <run>/right` |

### 3단계와 리플레이 결과 보기

**점 추적(CoTracker)으로 얻은 팔 궤적.** GOLD 펴기 시연 하나(10월 26일). 집게 끝 궤적은 자세를 어디서 얻었는지에
따라 색이 다르다: 초록 = AprilTag 마커 + CoTracker 점 + 깊이, 파랑 = 마커가 가려진 동안 CoTracker 점만, 주황 =
CoTracker 공백 구간의 IMU 회전. 왼쪽은 고정 카메라, 오른쪽은 같은 궤적을 로봇 base 좌표계에 그린 것이다.

![마커, CoTracker, IMU로 얻은 집게 끝 궤적](media/flatten_gold_trajectory_sources.gif)

**MASt3R-SLAM으로 공백을 채운 팔 궤적 (리플레이 시기).** 두 손의 집게 끝 궤적을 하나의 world 좌표계에 그렸다:
초록 = AprilTag 마커 + 깊이, 파랑 = 손목 어안 카메라의 MASt3R-SLAM(게이트를 통과한 구간만 채움), 주황 = IMU 회전 +
보간. 왼쪽은 고정 카메라, 오른쪽은 3D 궤적.

![마커, MASt3R-SLAM, IMU로 얻은 집게 끝 궤적](media/trajectory_reconstruction_large.gif)

**사람 시연의 로봇 리플레이.** 사람 시연(왼쪽)을 변환한 양팔 궤적을 reBot 팔에서 실행한 장면(오른쪽), run 200335,
같은 고정 카메라, 하나의 시계, 로봇 속도(0.35배)로 재생. 로봇이 시연 동작을 따라가고 집게가 셔츠 가장자리를 잡아
끈다. 리플레이만으로 접기까지 되지는 않았다([이력](docs/history_ko.md#개요-11월-2일-기준)).

![사람 시연 대 로봇 리플레이, 로봇 속도](media/replay3_200335_robot_time_94s_large.gif)

## 설치와 테스트

```bash
git clone https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation && cd egocentric-garment-manipulation
uv sync --group vision --group robot && uv run pytest -q   # Python 3.12, 테스트에는 torch(CPU면 충분)와 Pink IK가 필요
# CI와 같은 테스트 환경(Apple Silicon): pip install -r configs/laundry/envs/requirements-test.txt && pip install -e . --no-deps
python analysis/verify_claims.py                   # 리포트의 모든 숫자를 옮겨 온 문서와 대조: "227 claims checked, 0 failed."
python scripts/laundry/eval/replay_parity.py --synthetic          # 비교 채점기 자체 시험, 마지막 줄 "합성 시험 PASS"
python -m laundry_tracking.calib.robot_cam_solve --synthetic      # 시뮬레이션 데이터로 카메라 ↔ 로봇 풀이, 마지막 줄 "합성 시험 PASS"
```

테스트는 모든 패키지의 특성화 테스트(코드 구조를 바꾸기 전에 합성 녹화, 가짜 IK, golden 파일로 기록한 출력), 구조·import smoke
test, 합성 예제에 대한 참고 구현 회귀 검사입니다. torch, Pink IK, Flask가 있는 환경에서
598개 통과, 5개 건너뜀(lerobot 미설치, reBot URDF 미포함, 2026-11-12, 새로 만든 고정 버전 환경; 여기서는 CI가 건너뛰는 최적화
golden 두 개 중 `test_robotize_bi_take`도 실패). 기본 설치(2026-11-05)(`uv sync`나 `pip install -e ".[test]"`만)에서는
나머지가 모두 돌아서 425개 통과, 41개 건너뜀(torch, Pink, Flask가 필요한 테스트). CI는 `configs/laundry/envs/requirements-test.txt`의 고정 버전으로 Apple Silicon에서
돌고, IK 테스트(reBot URDF는 제조사 파일이라 포함하지 않음)와 다른 기계에서 출력이 1e-4 정도까지 달라지는 반복 최적화 golden 두 개를 건너뜁니다. dependency group: `vision`, `robot`, `train`, `collector`([코드 안내](docs/codebase_ko.md#설치-참고)).

**이 저장소만으로 재현할 수 없는 것:** 실제 시스템의 숫자는 포함하지 않은 녹화 데이터(원본 take, 학습된 checkpoint)에서
나왔습니다. reBot 제조사 플러그인과 URDF, 로봇 PC 드라이버, X-VLA 학습 패치는 포함하지 않았습니다. CoTracker와 학습은 GPU,
deploy는 로봇이 필요합니다. 숫자 검사는 기록과의 일관성을 보는 것이지 데이터에서 다시 계산하는 것이 아닙니다. 결과를 만든 환경:
[docs/environments.md](docs/environments.md).

## 결과

2026년 10월과 11월, 장치 하나와 시연자 한 명으로 측정했습니다. 전체 표와 지표 정의: [docs/results.md](docs/results.md)(영어)와
[테크니컬 리포트](report/report_ko.pdf). 11-05·11-07 숫자: 상태 노트 [11-05](docs/status_2026-11-05.md), [11-07](docs/status_2026-11-07.md)(영문). 리포트 뒤: [11-12](docs/status_2026-11-12.md).

| 단계 | 결과 |
|---|---|
| 캘리브레이션 | 카메라 ↔ 로봇 검증 재투영 0.98 px(옆 방향 약 1.3 mm), UMI 마커 리그 → 집게 끝 정지 1–4 mm(leave-one-out 1.8–4.8 mm) |
| 단일 팔 재현 | 이동 스트로크 16/18 통과, 축별 RMSE 중앙값 5.5–11.6 mm, 회전 스트로크 통과(오차 중앙값 0.3–1.6°), 명령 vs 사람 ≤ 3.4 mm이므로 남은 오차는 로봇 추종 |
| 사람 데이터로 첫 천 집기 | 오른팔이 시연의 집기 구간을 재생해 티셔츠를 집어 들어 올림 |
| 33초 데모의 양팔 전체 재생 (0.35배속, 94초) | 같은 계획으로 4회 실행의 반복 오차 약 1 mm, 집는 순간 명령 대비 2–8 mm(xy) |
| 사람 대비 오차 | 데모 전체 위치 RMSE: 재생한(로봇화된) 사람 궤적 대비 약 17 mm(오른팔) / 30 mm(왼팔), 원래 사람 궤적 대비 약 22 / 32–34 mm |
| 집는 높이 | 손가락 접촉을 허용한 뒤 집게가 테이블 위 14–16 mm(오른팔), 5–19 mm(왼팔)에서 닫힘(사람 약 10 mm), 명령 위치 대비 2–8 mm |
| 사람처럼 접기 | **아직 못 함.** 가장 좋은 실행에서 집게가 셔츠 가장자리를 잡아 끌었음. 성공률은 보고하지 않음 |
| 손목 카메라 SLAM을 위치 소스로 | **기각.** MASt3R-SLAM은 19–48 % 프레임만 유지(집는 순간 주변 0–38 %), ORB-SLAM3 mono-inertial은 스케일 발산 |
| CoTracker 점 추적 + 마커 + IMU (10-24/25) | held-out take에서 가린 마커 오차 중앙값 16 mm(왼) / 19 mm(오른), 마커만 쓰면 약 42 mm |
| 펴기 데이터셋 (10-26/27) | 성공 300 / 녹화 318, 앞 139개 등급 GOLD 133 / SILVER 5 / REJECT 1 |
| IK 이전 vs 이후 라벨 (10-26) | 앞 139개 시연(팔-프레임 약 123k): 집는 순간 IK → FK가 목표를 p95 약 58 mm(전체 53 mm) 옮김. IK 보정이 큰 프레임 20.3 %, 도달 불가능 0.8 % |
| 라벨의 테이블 여유 (10-29) | 라벨 프레임 44 %를 중앙값 27 mm(최대 65) 들어 올림 → 제거, 라벨 F = fusion 사람 집게 끝(집게 축, 0.25초 smoothing), 들어올림·clamp 없음 |
| 시연 속도 (10-29) | 사람 집게 끝 p95 0.93 m/s, 127 °/s — 로봇에 너무 빠름 → 0.3배속으로 늘린 데이터셋 |
| 손목 카메라 위치 (10-29) | 로봇 손목 카메라가 UMI보다 공구 축 쪽으로 11–13° 덜 기울고 25–44 mm 더 가까움 (테이블 태그 PnP) |
| 정책, 10-27 기준 | 아직 우리 데이터로 학습하지 않음. pretrain만 한 diffusion 정책(Soft-Fold pretrain DiT)이 로봇 dry run에서 안전 위반 없이 돎 |
| 로봇에서 돌린 정책 (10-28 →) | GOLD 데모 281개로 파인튜닝한 DiT·X-VLA를 시행 횟수를 세지 않은 로봇 시도에서 관찰: 천 쪽으로 가다가 닿기 전에 멈추고 일찍 닫음 — **성공률 없음** |
| 왼손 소매 하나 핀칭 (11-02) | 핀칭 시연 197개로 파인튜닝한 DiT(50 에포크 checkpoint): 11월 2일 실제 로봇 시행에서 집기 성공 관찰. 시행 횟수는 세지 않음 — **성공률 없음**. 놓친 뒤에는 다시 시도하지 않고 다른 곳을 잡음 |
| 펴기, FLAT250 (11-04/05) | val / test `val_score_mm` 71.67 / 72.27(시작) → **52.56 / 52.32**(13 에포크, 최고). 센 로봇 결과 없음 |
| 잡음 / 빈손 (11-05) | 손목 카메라 분류기, take 단위 5-fold CV, 이벤트별 F1 잡음 0.90, 빈손 0.78 |
| 펴기, FLAT-PGRASP 0.3배 (11-05 → 11-07) | train 301 / val 46 / test 45 시연; val / test `val_score_mm` 75.88 / 76.08(1 에포크) → **46.04 / 46.61**(53 에포크, 최고); held-out 셋이 달라 FLAT250과 비교할 수 없음 |
| 배포 지연 (11-06) | 측정한 명령→동작 지연: 팔 58 / 64 ms, 집게 173 / 171 ms(왼쪽 / 오른쪽). 카메라: 고정 1.4 ms, 손목 38 / 61 ms — 배포에서 맞춤 |
| 펴기 데이터로 X-VLA (11-10 →) | 우리 SoftFold 체크포인트(650k), domain 22: 5 에포크(train loss 0.6 → 0.07), 이어서 시연 330개로 100 에포크까지(11-12 학습 중). 오프라인·로봇 결과 아직 없음 |
| 태그 없는 위치 (11-11/12) | 태그 있는 take를 태그 없이 처리해 태그 측정과 비교: 집게 끝 오차 왼쪽 17 mm, 오른쪽 9–14 mm. 11-12 카메라 재조준은 depth 장면 ICP로 보정(`robot_cam_v007`, 보정 안 하면 6–27 mm) |
| 로봇에서 펴기 (11-06/07) | 로봇에서 0.3배가 1.0배보다 나아 보임(11월 6일). 11월 7일 두 집게가 셔츠 가장자리를 잡아 폈다가 다시 뭉침, 평평하게 내려놓지는 못함. 27번, 횟수를 세거나 채점하지 않음 — **성공률 없음** |

![실제 로봇 핀칭, 11월 2일](media/pinching_robot_20261102_113704.gif)

*2026년 11월 2일 11:37 실제 로봇 핀칭 실행(운영자가 성공으로 판단한 두 실행 중 하나, 시행 횟수는 세지 않음). 위: 고정 카메라와
정책의 다음 집게 목표, 아래: 두 손목 카메라 입력. 실제 시간의 약 2.8배 속도. 더 보기:
[이력](docs/history_ko.md#핀칭-10-30--11-02-로봇에서-첫-집기-성공-관찰-시행-횟수-세지-않음).*

**한계.** 성공률 없음: 재생은 측정과 영상으로 판단했고, 11월 2일 핀칭 집기를 포함한 정책 실행은 시행 횟수를 세지 않은 시행에서의
관찰입니다. 시연 하나만 양팔로 전체 재생했고, 약 1 mm 반복성은 결정성을 보여 줄 뿐 일반화는 아닙니다. 재생은 open loop입니다(천이
사람 시연 때와 같은 곳에서 시작해야 함). 상대 비교 지표는 상수 오프셋을 숨기므로 절대 위치는 줄자와 영상으로 확인했습니다. 전체
목록: [이력](docs/history_ko.md#한계).

## 바꾼 것과 이유 (짧은 결정 기록)

| 날짜 | 찾은 것 | 바꾼 것 |
|---|---|---|
| 10-21/23 | 손목 카메라 SLAM(MASt3R, ORB-SLAM3)이 하필 집는 순간 추적을 놓침 | 고정 카메라의 absolute 마커 + 깊이, 이어서 고정 카메라 점 추적이 위치 소스 |
| 10-24 | 마커가 절대 오차를 줄임. IMU 가속도를 넣으면 오히려 나빠짐 | fusion 변형 Cg: CoTracker + 마커, IMU 가속도는 끊긴 구간 안에서만 |
| 10-26 | IK → FK 라벨은 집는 순간 목표를 가장 많이 옮김 | 라벨은 IK 이전 Cartesian 포즈. IK는 오프라인에서는 검사기, 온라인에서는 실행기 |
| 10-29 | 테이블 여유 단계가 라벨 프레임 44 %를 들어 올림 | 라벨 F: fusion 사람 집게 끝, clamp·들어올림 없음. 제한은 deploy의 몫 |
| 10-29 | 사람 시연이 팔이 따라가기에 너무 빠름 | 시간을 늘인 데이터셋 복사본 |
| 10-29 | 영상 집게 라벨이 실제 집게와 다름 | UMI에 집게 서보를 토크 끈 엔코더로 다시 닮(`laundry_jaw.from_motor`) |
| 10-30 | 정책이 관측의 집게 상태를 그대로 따라 함 | 학습과 deploy에서 집게 상태를 가림(`GRIP_STATE_MASK=1`) |
| 11-04 | 시연 안의 점 추적 gap | trim 구간 안 가장 긴 gap으로 split, gated loss mask |
| 11-05 | 닫힌 집게만으로는 천을 잡았는지 모름 | 손목 카메라 잡음 라벨 → 상태의 집게 칸에 p_grasp(`PGRASP`) |
| 11-06 | 사람 데이터로 학습한 잡음 분류기가 로봇의 닫힌 빈 집게를 '잡음'으로 봄(0.97 / 0.94) | 로봇에서는 p_grasp = 집게 상태(닫힘 → 1), 로봇용 분류기가 생길 때까지 |
| 11-06 | 로봇에서 0.3배 정책이 1.0배보다 나아 보임(횟수 안 셈) | 펴기는 0.3배로 늘린 데이터 유지 |
| 11-06 | 로봇 집게가 늦게 닫힘(집게가 팔보다 약 110 ms 느림) | 배포에 UMI식 지연 맞추기 |
| 11-07 | IK가 관절 한계에 걸리면 팔이 비틀린 가장 가까운 자세로 감 | 그 팔은 마지막 관절값 유지 |
| 11-10 | 같은 데이터로 두 번째 정책 계열 | 우리 SoftFold 체크포인트의 X-VLA, 이어서 매일 데이터를 더하는 continual 학습 |
| 11-11 | 상대 action과 task-anchor 상태는 마커의 절대 위치가 필요 없음 | 태그 전부 제거. L 테이프 시작 표시, 정지 때의 IMU 방향, depth로 시작 시드 |

그 전의 결정(로봇 모델 보정, 제어기 모델, 자르지 않고 연속 변형)과 이유: [이력](docs/history_ko.md#어려웠던-점과-바꾼-것).
재생에서 펴기까지 날짜순 설명: [docs/history_ko.md](docs/history_ko.md).

## 하드웨어와 출처

> **HandUMI는 우리 설계가 아닙니다.** BrikHMP18(murobotics-ai)이 설계·공개했습니다(Apache-2.0):
> [murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw). [`hardware/umihand/`](hardware/umihand/README.md)는
> 원본 파일의 우리 제작 사본과 우리 변경분입니다.

![UMI 집게](media/umi_hands.jpg)

*손에 끼는 집게 두 개(왼쪽 / 오른쪽), 각각 fiducial 마커 3장(10-21 사진, 10-23 이전. 그 뒤로 손목 카메라 뒷면에 더 작은 30 mm
마커가 하나씩 더 있습니다).*

원본 부품을 출력·조립하고 우리 용도에 맞게 조금 바꿨습니다. VR 추적 대신 손목 IMU, 손마다 서로 다른 면에 AprilTag 여러 장(적어도
한 장은 외부 카메라를 향하도록, 10-23부터 손목 카메라 뒷면에 더 작은 네 번째 태그, 그 태그 하나의 평면 포즈 뒤집힘은 IMU로 해결),
controller-support 연결부·카메라 마운트·서보 커버 수정(`hardware/umihand/mods/`), 그리고 10-13부터 10-28까지 집게 서보 제거(집게
상태는 손목 영상으로 추정). 10-29에 서보를 집게 열림을 기록하는 토크 끈 엔코더로 다시 달았습니다(Feetech STS3215,
[파이프라인](docs/pipeline_ko.md#위치-복원과-검증-24단계) 5번). HandUMI 자체는 Universal Manipulation Interface(UMI, Chi et al.,
RSS 2024)를 따릅니다. [ATTRIBUTION.md](ATTRIBUTION.md) 참고.

로봇은 reBot B601-DM(평행 집게가 달린 6축 팔 두 대, 우리 MIT 브리지로 구동)이고, 제조사 LeRobot 플러그인과 URDF는 포함하지
않았습니다. 고정 카메라는 테이블 위(113 cm)의 Orbbec Gemini 336 RGB-D입니다. 전체 장치 목록: [파이프라인](docs/pipeline_ko.md#시스템-한눈에-보기).
이전 작업: [egocentric-cube-stacking](https://github.com/physical-ai-robot-learning/egocentric-cube-stacking), 관련 연구:
[이력](docs/history_ko.md#이전-작업과-관련-연구).

## 라이선스 · 감사의 말 · 인용

**라이선스.** 제품이 아니라 진행 중인 연구 프로젝트입니다. 숫자는 장치 하나, 시연자 한 명, 적은 실행 횟수에서 나왔고, 실패를
포함해 측정한 그대로 보고합니다. 이 저장소의 코드, 설정, 문서, 미디어는 **PolyForm Noncommercial License 1.0.0**([LICENSE](LICENSE))을
따릅니다. 연구, 교육, 개인 및 기타 비상업적 용도로는 자유롭게 쓸 수 있고, **상업적 사용은** 저자와 별도의 서면 합의 없이는
**허용되지 않습니다**. 예외: `hardware/umihand/`는 원본 HandUMI의 Apache-2.0 라이선스를 그대로 따릅니다
([LICENSE](hardware/umihand/LICENSE), [NOTICE](hardware/umihand/NOTICE.md)). 제3자의 설계와 방법은 각자의 라이선스를 따릅니다
([ATTRIBUTION.md](ATTRIBUTION.md)).

**감사의 말.** 손에 끼는 집게는 BrikHMP18의 HandUMI([murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw),
Apache-2.0)이고 UMI(Chi et al., RSS 2024)를 따릅니다. [하드웨어와 출처](#하드웨어와-출처)와 [ATTRIBUTION.md](ATTRIBUTION.md) 참고.
학습 계획에 적힌 pretrain 데이터: Soft-Fold, iPhUMI(MIT), PrimeBot UMI(CC BY-SA 4.0).

**출처 이력.** 2026-11-01까지 전체 코드와 세부 문서는 별도 작업 저장소 `egocentric-laundry-folding`에 있었고, 이 저장소는 공개용
정리본이었습니다. 두 저장소를 여기로 합쳤습니다. 11월 5일 코드 트리를 지금 구조의 유지 관리 코드로 바꿨습니다(동작은 같음, 옛 경로
→ 새 경로는 [docs/refactor_2026-11.md](docs/refactor_2026-11.md)). `report/archive/`에 보관한 테크니컬 리포트와 날짜별 상태 노트는
그 날짜의 경로를 적고 있습니다.

**인용.** [CITATION.cff](CITATION.cff)를 보세요(GitHub의 "Cite this repository").

```bibtex
@misc{lee_egocentric_garment_manipulation,
  author       = {Lee, Jeong Hwan},
  title        = {Egocentric Human Demonstrations for Bimanual Garment Manipulation},
  howpublished = {\url{https://github.com/physical-ai-robot-learning/egocentric-garment-manipulation}}
}
```

**저자.** **이정환 (Jeong Hwan Lee)** — Robotics Engineer, Physical AI / Robot Learning

관심 분야: 사람 시연으로 로봇 조작 학습, egocentric·착용형 데이터 수집, embodiment를 넘는 action 표현, 양손·변형 물체 조작,
vision-language-action 모델, 로봇 학습 데이터의 캘리브레이션과 평가 방법론.

GitHub: <https://github.com/physical-ai-robot-learning>
