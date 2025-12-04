TRAINER=CarPLAN
VAL_N_EPOCH=1
CACHE=carplan_1M_cache

# 학습
CONFIG=CarPLAN_1M
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py --version $CONFIG --batch 2 --cache $CACHE --epoch 25 --trainer $TRAINER --check_val_every_n_epoch $VAL_N_EPOCH --save_top_k 24 --CIL True