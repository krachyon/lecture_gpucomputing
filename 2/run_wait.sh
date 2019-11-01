#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o wait_out.txt

#for _ in {0..1} 
#do
bin/wait
#done
