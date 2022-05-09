#!/bin/sh -l
# FILENAME: dtt

#SBATCH -A partner 
#SBATCH --nodes=1 --gpus-per-node=2 --cpus-per-task=16
#SBATCH --time=1-00:00:00 
#SBATCH --job-name transfer

module load anaconda/5.3.1-py37
source activate basic
cd ..
#-- PGD attack dataset generator for tansfer attacks
seed_begin=1
seed_end=10
for gen_model in SENSE IAAT TRADE MART "MMA-12"
do

python attack_generator.py --generating_model $gen_model --seed_begin $seed_begin --seed_end $seed_end
done

#--attacking transfer attacks
seed_begin=1
seed_end=10
for gen_model in SENSE IAAT TRADE MART "MMA-12"
do
for def_model in SENSE MART TRADE 'MMA-12' IAAT 
do
python attack_applier.py --generating_model $gen_model --defense_model $def_model --seed_begin $seed_begin --seed_end $seed_end
done
done
