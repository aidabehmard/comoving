#!/bin/zsh -l
module load disBatch
sbatch -o logs/disbatch.o -e logs/disbatch.e -N8 --ntasks-per-node=64 --constraint=icelake -p cca -t 24:00:00 disBatch taskfile