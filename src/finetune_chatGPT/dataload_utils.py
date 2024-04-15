import pandas as pd 
import os
import glob

###########################################################################
# UTILS SCRIPT 1- DATA LOADING
# This utils scripts loads training and evaluation datasets. If your files are not labeled in the way we provide, you will need to change the names here.
# It is highly recommended to keep the data in the way we provided in the repository (ds_1_task_1_train_set and ds_1_task_1_eval_set), as otherwise you will need to change the names of training and evaluation sets many times in other scripts.
############################################################################


# Function 1 - Load train and evaluation datasets for finetuning
def load_train_and_eval_sets(data_dir: str, dataset_num: int, task_num: int, sample_size: int) \
        -> dict[str, pd.DataFrame]:
    datasets = dict()

    train_dataset_task_files = glob.glob(os.path.join(data_dir, f'ds_{dataset_num}__task_{task_num}_train_set*.csv'))
    eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set'
    datasets[eval_set_name] = pd.read_csv(os.path.join(data_dir, eval_set_name + '.csv'))

    if sample_size == 'all':
        train_dfs_ = {fn.strip('.csv'): pd.read_csv(fn) for fn in train_dataset_task_files}
        datasets.update(train_dfs_)
    else:
        train_df_fn = f'ds_{dataset_num}__task_{task_num}_train_set_{sample_size}'
        datasets[train_df_fn] = pd.read_csv(os.path.join(data_dir, train_df_fn + '.csv'))

        if train_df_fn not in [os.path.basename(fn).strip('.csv') for fn in train_dataset_task_files]:
            raise ValueError(f"Sample size {sample_size} not found for"
                             f" dataset {dataset_num} and task {task_num}")

    if dataset_num == 4:
        for df in datasets.values():
            df['text'] = df['title_h1'] + ' ' + df['text_200']

    return datasets
# Function 2 - Load full dataset only for zero-shot prediction
def load_full_dataset(data_dir: str, dataset_num: int, task_num: int) \
        -> dict[str, pd.DataFrame]:
    datasets = dict()
    eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set_full'
    datasets[eval_set_name] = pd.read_csv(os.path.join(data_dir, f'ds_{dataset_num}__task_{task_num}_full__for_zero_shot_classification.csv'))

    if dataset_num == 4:
        for df in datasets.values():
            df['text'] = df['title_h1'] + ' ' + df['text_200']

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

