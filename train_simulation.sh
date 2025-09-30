TRAINER=CarPLAN
VAL_N_EPOCH=1
CACHE=carplan_1M_cache

# 학습
CONFIG=CarPLAN_1M
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py --version $CONFIG --batch 2 --cache $CACHE --epoch 25 --trainer $TRAINER --check_val_every_n_epoch $VAL_N_EPOCH --save_top_k 24 --CIL True

# 시뮬레이션
CHALLENGE=CLS_NR
CUDA_VISIBLE_DEVICES=0 python run_simulation.py --version $CONFIG --challenge $CHALLENGE --threads 40 --ckpt 24 --filter test14_hard

CHALLENGE=CLS_R
CUDA_VISIBLE_DEVICES=0 python run_simulation.py --version $CONFIG --challenge $CHALLENGE --threads 40 --ckpt 24 --filter test14_hard