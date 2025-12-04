CONFIG=CarPLAN_1M

# 시뮬레이션
CHALLENGE=CLS_NR
CUDA_VISIBLE_DEVICES=0 python run_simulation.py --version $CONFIG --challenge $CHALLENGE --threads 40 --ckpt 24 --filter test14_hard

CHALLENGE=CLS_R
CUDA_VISIBLE_DEVICES=0 python run_simulation.py --version $CONFIG --challenge $CHALLENGE --threads 40 --ckpt 24 --filter test14_hard