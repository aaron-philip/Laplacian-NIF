#!/bin/bash

# both needed for tensorflow!
module load GCCcore/12.2.0
module load CUDA/11.8.0


source ~/.bashrc
conda deactivate
conda activate tddft-emulation
export PYTHONPATH=/mnt/scratch/philipaa/tddft-emulation/nif
ulimit -s unlimited
#python train_50_4_100_3_10.py
#python init_derivatives_dataset.py
python train_6sensor-50_4_50_4_5.py
