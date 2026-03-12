#First Executable Line
source ~/.bashrc
conda activate py

for seed in 0 1 2 3 4
do
  echo "Iteration seed $seed "
  for beta in 20
  do
    echo "Iteration beta $beta "
     python main.py  --exp-name='alexr2'  --beta=$beta --seed=$seed   \
            --loss='alexr2' --total-epochs=60 --kappa=0.005  --lr=1e-2 \
            --dataset='adult'  --th-start=-3 --th-end=3.1 --th-step=1  --scaling=1 \
            --lam=2e-4 --mmt=0.1 --nu=0.01 --outlr=0.002 --gamma=0.8 --gamma_p=0.1 --in_iter=5 --restart 
  done
done