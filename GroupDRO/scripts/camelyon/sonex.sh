source ~/.bashrc
conda activate py


# use seed 1, 2, 3
python main_came.py --alg SONEX --gpu 0 --lr 1e-3 --beta 0.1 --gamma 0.2 --lamda 0.1 --theta 0.05 \
    --lr_c 0.1 --seed 1 --epoch 10 --alpha 0.15 