## 1. CarPLAN 코드 다운로드

```bash
# 특정 디렉토리 (예: /home/jyyun/workshop)에서
git clone https://github.com/JunyongYun-SPA/HMG_SNU_CarPLAN.git
```

---

## 2. 데이터 다운로드

nuPlan v1.1 Dataset 다운로드 (https://www.nuscenes.org/nuplan)

- nuPlan Train Split
- nuPlan Test Split
- nuPlan Val Split

```bash
# 경로 예시
/home/jyyun/workshop/HMG_SNU_CarPLAN/dataset
```

---

## 3. 도커 이미지 실행

```bash
docker login
docker pull junyongyun/nuplan:latest

docker run -it --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/jyyun/workshop/HMG_SNU_CarPLAN:/home/jyyun/workshop/HMG_SNU_CarPLAN \
  -e DISPLAY=unix$DISPLAY \
  --ipc=host \
  --name nuplan {이미지ID}

# 컨테이너 실행
docker start nuplan
docker exec -it nuplan /bin/bash

# conda 환경 진입
conda activate nuplan
```

---

## 4. nuPlan 환경설정

도커 내부 `~/.bashrc`에 아래 환경변수 추가:

```bash
export NUPLAN_DATA_ROOT="/home/jyyun/workshop/HMG_SNU_CarPLAN/dataset"
export NUPLAN_MAPS_ROOT="/home/jyyun/workshop/HMG_SNU_CarPLAN/dataset/maps"
export NUPLAN_EXP_ROOT="/home/jyyun/workshop/HMG_SNU_CarPLAN/dataset/exp"
export NUPLAN_DEVKIT_ROOT="/home/jyyun/workshop/HMG_SNU_CarPLAN/nuplan-devkit"
```

---

## 5. 패키지 설치

```bash
pip install --upgrade pip==23.1

cd /home/jyyun/workshop/HMG_SNU_CarPLAN/nuplan-devkit/
pip install -e .
pip install -r ./requirements.txt

cd ..
sh ./script/setup_env.sh

cd /home/jyyun/workshop/HMG_SNU_CarPLAN/tuplan_garage/
pip install -e .
```

---

## 6. 데이터 전처리

```bash
# /home/jyyun/workshop/HMG_SNU_CarPLAN 에서 실행
sh ./script/preprocess/preprocess_1M.sh
```

또는

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python ./run_preprocessing.py \
  py_func=cache \
  +training=train_carplan \
  scenario_builder=nuplan \
  cache.cache_path=./cache/carplan_1M_cache \  # 캐시파일 경로 지정
  cache.cleanup_cache=true \
  scenario_filter=training_scenarios_1M \
  worker.threads_per_node=40
```

---

## 7. 학습 및 시뮬레이션

```bash
# /home/jyyun/workshop/HMG_SNU_CarPLAN 에서 실행
sh train_simulation.sh   # 해당 파일 참고
```

---

## 8. 주요 코드

### 데이터 전처리
- **carplan_feature_builder.py**
  - `_build_feature` : 학습에 필요한 데이터 전처리 수행

### 학습
- **carplan_model.py**
  - CarPLAN 전체 모델 코드
- **planning_decoder_deepseek_v2.py**
  - CarPLAN의 Decoder 코드

### 시뮬레이션
- **carplan_planner.py**
  - `compute_planner_trajectory` : CarPLAN 시뮬레이션 수행

---

## Reference
- [nuPlan Dataset](https://www.nuscenes.org/nuplan)
