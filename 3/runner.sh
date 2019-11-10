#!/usr/bin/env bash
#SBATCH --gres=gpu
#SBATCH -o result.csv

python3 runner.py
