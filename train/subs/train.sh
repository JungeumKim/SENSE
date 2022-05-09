#!/bin/sh -l
# FILENAME: dtt

#SBATCH -A training 
#SBATCH --nodes=1 --gpus-per-node=4 --cpus-per-task=20
#SBATCH --time=7-00:00:00 
#SBATCH --job-name train_wide&small

module load anaconda/5.3.1-py37
source activate basic

cd ../..

python main.py --lossCut 0.5  --cfg_path 'train/train_sense.json' 
python main.py --lossCut 0.7  --cfg_path 'train/train_sense.json' --nepochs 250
python main.py --lossCut 0.1  --cfg_path 'train/train_sense.json' --nepochs 200
python main.py --lossCut 0.3  --cfg_path 'train/train_sense.json' --nepochs 200



for capa in 1 2 3 4 
do 
python main.py --capa $capa --cfg_path 'train/train_sense_capa.json' 
done
