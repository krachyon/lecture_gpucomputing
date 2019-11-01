#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o ex2_out.txt

for _ in {0..20} 
do
	bin/nullKernelAsync
done
