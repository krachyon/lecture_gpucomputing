#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o result.txt

python3 runner.py
