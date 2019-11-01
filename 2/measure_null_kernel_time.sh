#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o nullkernel_timings_out.txt

for _ in {0..1} 
do
	bin/nullKernelAsync
done
