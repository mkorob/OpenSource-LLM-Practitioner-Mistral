#!/bin/bash
#SBATCH  --output=compute_jobs_log/mistral_ds1_t1_500.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:A100:1

module load multigpu
module load mamba
source activate llm_finetune_mistral_v2
export PYTHONPATH=$PYTHONPATH:/scratch/mkorob/OpenSource-LLM-Practitioner-Mistral/src/utils

python src/MISTRAL/01-finetune_MISTRAL_PeFT.py \
--dataset 1 \
--task 1 \
--sample_size 500
