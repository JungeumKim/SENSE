#!/bin/sh -l
# FILENAME: dtt

#SBATCH -A partner
#SBATCH --nodes=1 --gpus-per-node=2 --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name large_ev

module load anaconda/5.3.1-py37
source activate basic
cd ..

for capa in 1 2 3 4
do
model_pth=../trained_models/smaller_nets/capa$capa/checkpoint.pth 

python Evaluation_small_models.py --eval_group2 --model_path $model_pth --capa $capa 
done

for lossCut in 0.1 0.3 0.5 0.7
do
model_pth=../trained_models/wider_net/c$lossCut/checkpoint.pth 
python evaluating_trained_models.py --eval_group2 --model_path $model_pth 

done
