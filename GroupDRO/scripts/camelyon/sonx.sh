source ~/.bashrc
conda activate py



python main_came.py --alg SONX --gpu 0 --lr 1e-3 --gamma 0.2 --theta 0.1 \
    --lr_c 0.1 --seed 1 --epoch 10 --alpha 0.15
