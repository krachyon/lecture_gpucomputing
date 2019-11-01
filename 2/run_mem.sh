#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o memory.txt

#for _ in {0..1} 
#do
bin/memory
#done
