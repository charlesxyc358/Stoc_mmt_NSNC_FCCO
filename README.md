<h1> Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization</h1>

This repository provide codes for the paper [Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](https://arxiv.org/abs/2506.02504) .

## Installation

```bash
# Clone the repository
git clone https://github.com/charlesxyc358/Stoc_mmt_NSNC_FCCO
cd Stoc_mmt_NSNC_FCCO
conda create -n fcco python==3.11
conda activate fcco
pip install -r requirements.txt
```

## GroupDRO
#### First navigate to GroupDRO folder
```
cd GroupDRO
```
#### Download and Prepare the (labeled)Camelyon17_v1.0 and Amazon_v2.1 datasets
Download from [Wilds Datasets](https://worksheets.codalab.org/worksheets/0xb44731cc8e8a4265a20146c3887b6b90), then put the downloaded tar.gz files under ./data folder and extract to the corresponding folder:
```
mkdir data
cd data
mkdir camelyon17_v1.0
tar -xzf ./data/camelyon17_v1.0.tar.gz -C ./data/camelyon17_v1.0
mkdir amazon_v2.1
tar -xzf ./data/amazon_v2.1.tar.gz -C ./data/amazon_v2.1
cd ..
```
#### Train on Camelyon17 dataset with SONEX
```
python main_came.py --alg SONEX --gpu 0 --lr 1e-3 --beta 0.1 --gamma 0.2 --lamda 0.1 --theta 0.05 \
    --lr_c 0.1 --seed 1 --epoch 10 --alpha 0.15
```
#### Train on Amazon dataset with SONEX
```
python main_amz.py --alg SONEX --gpu 0 --lr 2e-5 --beta 0.1 --gamma 0.2 --lamda 0.1 --theta 0.01 \
    --wd 1e-2 --lr_c 0.1 --seed 1 --epoch 4 --alpha 0.15 
```

## AUC Maximization with ROC Fairness Constraints
#### First navigate to AUCMax folder
```
cd AUCMax
```
#### Download and Prepare the adult and compas datasets 
Run the following command to download the datasets.
```
mkdir data
mkdir data/adult
mkdir data/compas
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O data/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O data/adult/adult.test
wget https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores.csv -O data/compas/compas-scores.csv
```

#### Train on Adult dataset with Alexr2
```bash
sh ./scripts/alexr2.sh
```
#### Train on Compas dataset with Alexr2
```bash
sh ./scripts/compas_alexr2.sh
```

## Citation
If you find this tutorial helpful, please cite our paper:
```
@article{chen2025stochastic,
  title={Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization},
  author={Chen, Xingyu and Wang, Bokun and Yang, Ming and Lin, Qihang and Yang, Tianbao},
  journal={arXiv preprint arXiv:2506.02504},
  year={2025}
}
```
