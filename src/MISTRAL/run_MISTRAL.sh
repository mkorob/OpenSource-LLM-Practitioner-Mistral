#!/bin/bash
#SBATCH  --output=compute_jobs_log/llama3_zeroshot_view.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:A100:1

module load multigpu
module load mamba
source activate llm_finetune_v2
export PYTHONPATH=$PYTHONPATH:/scratch/mkorob/OpenSource-LLM-Practitioner-Guide-llama3

python src/MISTRAL/01b-few_zeroshot_MISTRAL_PeFT_script.py