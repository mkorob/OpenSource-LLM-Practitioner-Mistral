#!/bin/bash
#SBATCH --output=compute_jobs_log/all_out.out

# Define the combinations
declare -a datasets=("2" "3" "4")
declare -a tasks=("1" "2", "3")

# Define sample sizes for specific dataset-task combinations
declare -A sample_sizes
sample_sizes["2_1"]="50 100 250 500 1000"
sample_sizes["2_2"]="50 100 250 500"
sample_sizes["3_1"]="50 100 250 500 1000 1500"
sample_sizes["3_3"]="50 100 250 500"
sample_sizes["4_1"]="50 100 250 500 1000 1500"
sample_sizes["4_2"]="50 100 250 500 1000"

# Create job scripts directory if it doesn't exist
mkdir -p job_scripts

# Loop through the combinations and create job scripts
for dataset in "${datasets[@]}"; do
    for task in "${tasks[@]}"; do
        key="${dataset}_${task}"
        if [ -n "${sample_sizes[$key]}" ]; then
            for sample_size in ${sample_sizes[$key]}; do
                job_script="job_scripts/job_ds${dataset}_t${task}_s${sample_size}.sh"
                
                # Create the job script from the template
                cat <<EOT > $job_script
#!/bin/bash

#SBATCH --output=compute_jobs_log/mistral_ds${dataset}_t${task}_s${sample_size}_%j.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --constraint=GPUMEM80GB
#SBATCH --gres=gpu:A100:1

module load multigpu
module load mamba
source activate llm_finetune_mistral_v2
export PYTHONPATH=\$PYTHONPATH:/scratch/mkorob/OpenSource-LLM-Practitioner-Mistral/src/utils

python src/MISTRAL/01-finetune_MISTRAL_PeFT.py \
--dataset ${dataset} \
--task ${task} \
--sample_size ${sample_size}
EOT
                
                # Submit the job script
                sbatch $job_script
            done
        else
            echo "No sample sizes available for dataset ${dataset}, task ${task}"
        fi
    done
done