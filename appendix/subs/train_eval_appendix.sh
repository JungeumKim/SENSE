#!/bin/sh -l
# FILENAME: dtt

#SBATCH -A partner 
#SBATCH --nodes=1 --gpus-per-node=2 --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name sense-capa0adv

module load anaconda/5.3.1-py37
source activate basic

cd ..

#nat 
python main_appendix.py --cfg_path configs/train_sense_capa.json --lossCut 1.0 --capa 0 
python main_appendix.py --cfg_path configs/train_sense_capa.json --lossCut 1.0 --capa -1 

#adv 
python main_appendix.py --cfg_path configs/train_R_AT_capa.json --capa 0 
python main_appendix.py --cfg_path configs/train_R_AT_capa.json --capa -1 

#sense
for capa in -1 0 
do 
for lossCut in 0.1 0.5 0.7
do 

python main_appendix.py --cfg_path configs/train_sense_capa.json --lossCut $lossCut --capa $capa 

done
done

head ./results/capa*/full/*/*/*/*/*/eval*.json



