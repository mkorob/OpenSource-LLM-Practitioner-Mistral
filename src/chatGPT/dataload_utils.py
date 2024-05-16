import pandas as pd 
import os
import glob

###########################################################################
# UTILS SCRIPT 1- DATA LOADING
# This utils scripts loads training and evaluation datasets. If your files are not labeled in the way we provide, you will need to change the names here.
# It is highly recommended to keep the data in the way we provided in the repository (ds_1_task_1_train_set and ds_1_task_1_eval_set), as otherwise you will need to change the names of training and evaluation sets many times in other scripts.
############################################################################


# Function 1 - Load train and evaluation datasets for finetuning
def load_train_and_eval_sets(data_dir: str, train_set_name: str, eval_set_name: str) \
        -> dict[str, pd.DataFrame]:
    datasets = dict()

    datasets[eval_set_name] = pd.read_csv(os.path.join(data_dir, eval_set_name + '.csv'))
    datasets[train_set_name] = pd.read_csv(os.path.join(data_dir, train_set_name + '.csv'))

    return datasets


# Function 2 - Load full dataset only for zero-shot prediction
def load_full_dataset(data_dir: str, dataset_name: str) \
        -> dict[str, pd.DataFrame]:
    datasets = dict()
    datasets[dataset_name] = pd.read_csv(os.path.join(data_dir, f'{dataset_name}.csv'))

    return datasets


def load_dataset_task_prompt_mappings(dataset_num, task_num, dataset_task_mappings_fp):

    dataset_task_mappings = pd.read_csv(dataset_task_mappings_fp, encoding='unicode_escape')

    dataset_idx = dataset_task_mappings.index[
        (dataset_task_mappings["dataset_number"] == dataset_num) & (dataset_task_mappings["task_number"] == task_num)]

    if len(dataset_idx) == 0:
        raise ValueError("Invalid dataset-task combination")

    elif len(dataset_idx) > 2:
        raise ValueError("Multiple dataset-task combinations found")
    else:
        dataset_idx = dataset_idx[0]

    return dataset_idx, dataset_task_mappings

