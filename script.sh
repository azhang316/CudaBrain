#!/bin/bash
#SBATCH	--job-name=cudabrain
#SBATCH --output=cudabrain_%j.out
#SBATCH --error=cudabrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=azhang36@ucsc.edu
#SBATCH --partition=am-148-s20
#SBATCH --partition am-148-s20
#SBATCH --qos.am-148-s20
#SBATCH --account=am-148-s20

module load cuda10.1/10.1.168

srun cudabrain.exe
