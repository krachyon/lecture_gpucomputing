sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 1
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 2
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 4
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 8
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 16
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 32
sbatch -p hpc --array=1-2 ./run.sh cuda_naive 1 2501 1 1 64

sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 1
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 2
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 4
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 8
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 16
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 32
sbatch -p hpc --array=1-2 ./run.sh cuda_shared 1 2501 1 1 64

