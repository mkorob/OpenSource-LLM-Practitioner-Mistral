# %% [markdown]
# # Template for Zero-shot and Few-shot Classification with LLaMA through Low-Ranking Adapters

# %% [markdown]
# ### Import all Modules

# %%
#IMPORTANT - to ensure package loading, first add the path of the utils folder to your system path
import os
import sys

#In Python - this is going to be the location of the folder, not the repository
module_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(module_dir, "src" "utils")))

# %%
import tqdm
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig)
import pandas as pd
import torch
import numpy as np
import random
import wandb
import argparse

from dataload_utils import load_full_dataset, load_dataset_task_prompt_mappings

# %% [markdown]
# ### Setup Arguments and Data

# %% [markdown]
#  In the following code block, you are asked to set up several key parameters that will define the behavior and environment of your fine-tuning process:
# 
# 1. **WandB Project Name (`WANDB_PROJECT_NAME`)**: This is the name of the project in Weights & Biases (WandB) where your training run will be logged. WandB is a tool that helps track experiments, visualize data, and share insights. By setting the project name here, you ensure that all the metrics, outputs, and logs from your training process are organized under a single project for easy access and comparison. Specify a meaningful name that reflects the nature of your training session or experiment. If you leave the argument empty, the project will not be tracked on WandB.
# 
# 2. **Model Name (`MODEL_NAME`)**: Here, you select the size of LLAMA model that you wish to fine-tune. This notebook was ran and tested on (`meta-llama/Llama-2-70b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `OASST-LLAMA 30b` (not available on HuggingFace anymore)).
# 

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, required=True, help="Type of task to run inference on",
                    choices=[1, 2, 3, 4, 5, 6])
parser.add_argument("--dataset", type=int, required=True, help="Dataset to run inference on",
                    choices=[1, 2, 3, 4])

args = parser.parse_args()


# %%
WANDB_PROJECT_NAME = "llama3_annotations_llm_comparison"
# Name of the model to finetune (this script was tested on LLAMA-2 70b, LLAMA-2 13b, and OASST-LLAMA 30b)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# %% [markdown]
# In order to run LLAMA-2 models, you need to register yourself at the HuggingFace model page (https://huggingface.co/meta-llama/Llama-2-70b-chat-hf). Then, you can either insert the token here (not recommended if sharing a repository on GitHub), or input it in the hf_token.txt as done here and ensure it is included in the .gitignore.
with open(os.path.join(module_dir, "src", "MISTRAL", "hf_token.txt"), "r") as file:
    hf_token = file.read().strip()
# %%

# %% [markdown]
# This is an optional parameter to run if your default transformers cache location does not contain enough storage to load the LLAMA models. Otherwise, you can keep it as is.

# %%
#cache_location = os.environ['HF_HOME']
cache_location = "../cache"

os.environ['TRANSFORMERS_CACHE'] = cache_location
os.environ['HF_HOME'] = cache_location

# %% [markdown]
# In the next code block, you are required to set up various configuration variables that will dictate how the inference processes are executed. These variables are crucial as they define the nature of the task, the data, and the specific behaviors during the model's training and evaluation.
# 
# 1. **Task (`task`)**: Specify the type of task you want to run inference on. The task is represented by an integer, with each number corresponding to a different type of task (e.g., 1, 2, 3, etc.). You must select from the predefined choices, which are typically mapped to specific NLP tasks or scenarios.
# 
# 2. **Dataset (`dataset`)**: Choose the dataset on which you want to run inference. Like tasks, datasets are identified by integers, and each number corresponds to a different dataset. Ensure that the dataset selected is relevant to the task at hand.
# 
# 3. **Output Directory (`output_dir`)**: Define the path to the directory where you want to store the generated samples. This is where the output of your training and inference processes will be saved.
# 
# 4. **Model Directory (`model_dir`)**: Define the path to the directory where you want to store the generated models. This should have sufficient memory.
# 
# 5. **Llama-2 Prompt** (`use_llama2_prompt`)**: Keep True if running LLAMA-2 models (provides correct prompts with system and user message separated). Set False if running OASST models.
# 
# 6. **Division Line for User Message** (`system_user_prompt_division_line`)**: Length in number of lines of the user message. Relevant only for LLAMA-2 models.
# 
# 7. **Random Seed (`seed`)**: Setting a random seed ensures that the results are reproducible. By using the same seed, you can achieve the same outcomes on repeated runs under identical conditions.
# 
# 8. **Data Directory (`data_dir`)**: Specify the path to the directory containing the datasets you plan to use for training and evaluation.
# 
# 9. **Label Usage (`not_use_full_labels`)**: This boolean variable determines whether to use the full label descriptions or abbreviated labels during training and inference. Setting it to `False` means full labels will be used.
# 
# 10. **Dataset-Task Mappings File Path (`dataset_task_mappings_fp`)**: Define the path to the file containing mappings between datasets and tasks. This file is crucial for ensuring the correct dataset is used for the specified task.
# 
# 11. **Maximum prompt length (`max_prompt_len`)**: The maximum length of prompt in tokens to be taken as input before truncating the input. Longer input sequences require more computational power to run, so the shortest sequence required to capture the text is recommended.
# 
# 12. **Few shot indicator (`few_shot`)**: Set to True to run few-shot learning with the prompt defined as few_shot_prompt in the dataset mapping dataframe.
# 
# 13. **Batch size (`batch-size`)**: Number of observations used in each training and validation batch. Larger batch size requires more computational memory as one batch needs to fit on one machine, but makes learning more stable. We found that for FLAN-XL, batch size of 8 was possible by taking batch size of 4 and accumulating results of 2 batches (see  (`gradient_accumulation_steps`) below)
# 
# 14. **Gradient accumulation steps (`gradient_accumulation_steps`)**: In a case where gradient accumulation steps is larger than 1, instead of updating the gradient after each batch, the gradient is updated after the sum of _n_ batches. This allows to train a model to learn on a larger global batch (_batch size_ * _gradient accumulation steps_) than the one that is able to fit on one machine.
# 
# 15. **Run_name (`run_name`)**: Optional run_name to store in WandB. We recommend to keep it null, which generates an automatic run name based on all the relevant parameters of finetuning for easier tracking. 
# 
# 16. **Maximum output length (`max_new_tokens`)**: Maximum length of prediction produced by LLAMA. For data labelling, it does not make sense to make it longer than that. 
# 
# 17. **LoRA hyperparameters:** The values provided below were taken from Stanford Alpaca LoRA repository: https://github.com/tloen/alpaca-lora/blob/main/finetune.py.
# 

# %%
# Configuration Variables

# Type of task to run inference on
task = args.task  # 

# Dataset to run inference on
dataset = args.dataset  #

# Path to the directory to store the generated predictions
output_dir = 'data'

# Path to the directory to store the models (make sure this location is included in the .gitignore if using GitHub)
model_dir = 'data'

#If using LLAMA2, keep True. Set False only for OASST-LLAMA.
llama_type = "mistral"

#This is relevant for LLAMA2, where the system and user message are separated by context tokens. You should count how many lines your user message takes (in this case, 3)
system_user_prompt_division_line = 3

# Random seed to use
seed = 2019

# Path to the directory containing the datasets
data_dir = 'data'

# Whether to use the full label
not_use_full_labels = False

# Path to the dataset-task mappings file
dataset_task_mappings_fp = os.path.normpath(os.path.join(module_dir, 'dataset_task_mappings.csv'))

#Maximum length of prompt to be taken by the model as input (check documentation for current maximum length)
max_prompt_len = 4096

#Zero or few-shot binary variable
few_shot = False

#run name - Optional Argument if you want it to be called something else than the default way (defined below)
run_name = ""

#maximum length of sequence to produce
max_new_tokens = 100

#Text Generation parameters (Values below taken from Stanford Alpaca LoRA repository : https://github.com/tloen/alpaca-lora/blob/main/generate.py)
temp = 0.05                                     
top_p = 0.75      
top_k = 40

# %%
dataset_name = f'ds_{dataset}__task_{task}_full__for_zero_shot_classification'

# %% [markdown]
# **Customizing for Your Own Tasks:**
# If you plan to run a custom task or use a dataset that is not predefined, you will need to make modifications to the `label_utils` file. This file contains all mappings for different datasets and tasks. Adding your custom task or dataset involves defining the new task or dataset number and specifying its characteristics and mappings in the `label_utils` file. This ensures that your custom task or dataset integrates seamlessly with the existing framework for training and inference.

# %% [markdown]
# ### Define Utility Functions

# %%
def set_all_seeds(seed: int = 123):
    # tf.random.set_seed(123)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seed with the `transformers` library
    # set_seed(seed)

# %% [markdown]
# ## Main Implementation

# %%
set_all_seeds(seed)

# %%
exp_name = run_name if run_name != '' else f'{MODEL_NAME}_ds_{dataset}_task_{int(task)}_sample_{0}_prompt_max_len_{max_prompt_len}'
if few_shot:
    exp_name += "few_shot"
exp_name = exp_name.replace('.', '_')

# Initialize the Weights and Biases run
if WANDB_PROJECT_NAME != "":
    wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT_NAME,
        name=exp_name,
        # track hyperparameters and run metadata
        config={
            "model": MODEL_NAME,
            "dataset": dataset,
            "task": task,
            "max_prompt_len": max_prompt_len
        }
    )

# %%
if not_use_full_labels:
    exp_name += '_label_abbreviation'
    labelset_col = 'labelset'
else:
    labelset_col = 'labelset_fullword'

# %%
print('Running exp:', exp_name)

# %% [markdown]
# ### Load Data and the prompt

# %%
prompt_col = 'few_shot_prompt' if few_shot else 'zero_shot_prompt'
# Load the prompt
dataset_idx, dataset_task_mappings = load_dataset_task_prompt_mappings(
    dataset_num=dataset, task_num=task, dataset_task_mappings_fp=dataset_task_mappings_fp)

# Get information specific to the dataset
label_column = dataset_task_mappings.loc[dataset_idx, "label_column"]
labelset = dataset_task_mappings.loc[dataset_idx, labelset_col].split("; ")
labelset = [label.strip() for label in labelset]
prompt = dataset_task_mappings.loc[dataset_idx, prompt_col]

# Get the system or instruction prompt and the user prompt format
system_prompt = ('\n'.join(prompt.split('\n')[:-system_user_prompt_division_line])).strip()
user_prompt_format = ('\n'.join(prompt.split('\n')[-system_user_prompt_division_line:])).strip()

# Log the system prompt and user_prompt_format as files in wandb
if WANDB_PROJECT_NAME != "":
    prompts_artifact = wandb.Artifact('prompts', type='prompts')
    with prompts_artifact.new_file('system_prompt.txt', mode='w') as f:
        f.write(system_prompt)
    with prompts_artifact.new_file('user_prompt_format.txt', mode='w') as f:
        f.write(user_prompt_format)
    wandb.run.log_artifact(prompts_artifact)

# Load the train and eval datasets with the full prompt format
print(f'label_columns: {label_column}')
print(f'labelset: {labelset}')

datasets = load_full_dataset(
    data_dir=data_dir, dataset_name=dataset_name, task_num = task,
    label_column=label_column, labelset=labelset, full_label=not not_use_full_labels, system_prompt=system_prompt, user_prompt_format=user_prompt_format,
    llama_type=llama_type)


# %%
print(f"Eval set example with completion ({len(datasets['eval'])} rows): ")
print("-" * 50 + '\n')
#print(datasets["eval"]["text"][0])
print('\n\n')

print(f"Eval set without completion ({len(datasets['eval_wo_completion'])} rows): ")
print("-" * 50 + '\n')
#print(datasets["eval_wo_completion"]["text"][0])
print('\n\n')


# %% [markdown]
# ### Define the model, tokenizers, data collator

# %%
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left", cache_dir = cache_location, token = hf_token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, cache_dir = cache_location, token = hf_token)

# %% [markdown]
# ### Run Predictions

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left", token = hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Default params from alpaca-lora generate script (commonly used)
generation_config = GenerationConfig(
    temperature=temp,
    top_p=top_p,
    top_k=top_k,
    do_sample=True,
    max_new_tokens=max_new_tokens,
    pad_token_id = tokenizer.pad_token_id
)

with torch.no_grad():
    predictions_out = []
    for i, input_text_i in tqdm.tqdm(enumerate(datasets["eval_wo_completion"]["text"])):
        # Tokenize the text
        tokenized_text_i = tokenizer(
            text_target=input_text_i,
            padding=False,
            max_length=max_prompt_len,
            truncation=True,
            return_tensors="pt"
        )

        # Generate the completions
        outputs = model.generate(
            input_ids=tokenized_text_i["input_ids"].cuda(),
            generation_config=generation_config
        )

        generated_text_minibatch = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        predictions_out += generated_text_minibatch

        if i == 0:
            print("Sample prediction: ")
            print(predictions_out[0])

# %%
os.path.join(output_dir, 'predictions', MODEL_NAME.replace("/", "_"))

# %%
#load csv to add the predictions on 
predictions_dir = os.path.join(output_dir, 'predictions', MODEL_NAME.replace("/", "_"))
os.makedirs(predictions_dir, exist_ok=True)
eval_df = pd.read_csv(os.path.join(data_dir, f"{dataset_name}.csv"))
eval_df['prediction'] = predictions_out

# %%
os.path.join(predictions_dir, f'{exp_name.replace("/", "_")}.csv')

# %%
#export to csv
eval_df.to_csv(os.path.join(predictions_dir, f'{exp_name.replace("/", "_")}.csv'))

# %% [markdown]
# ### Terminate WandB

# %%
if WANDB_PROJECT_NAME != "":
    wandb.finish()

# %%



