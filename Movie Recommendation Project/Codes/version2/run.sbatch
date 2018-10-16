#!/bin/bash
#
#SBATCH --job-name=Zhao_Swan
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --time=05:00:00
 
module purge
module load torchvision/python3.5/0.1.9
 
cd /home/sy2160/AML
python3 trainAEKM.py > example.out 2>&1