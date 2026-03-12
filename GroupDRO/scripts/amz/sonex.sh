source ~/.bashrc
conda activate py


# use seed 1, 2, 3
python main_amz.py --alg SONEX --gpu 0 --lr 2e-5 --beta 0.1 --gamma 0.2 --lamda 0.1 --theta 0.01 \
    --wd 1e-2 --lr_c 0.1 --seed 1 --epoch 4 --alpha 0.15 
