#!/bin/sh
#SBATCH --job-name=resnet
#SBATCH --time=120
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=casanath@comp.nus.edu.sg

TMPDIR=`mktemp -d`

mkdir $TMPDIR/data

cp -r ~/bt5153/data/* $TMPDIR/data/

srun python predict_resnet.py $TMPDIR

rm -rf $TMPDIR