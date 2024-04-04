from datasets import Dataset, DatasetDict
import pandas as pd 
import os
import glob
from src.utils import map_label_to_completion

def load_train_and_eval_datasets(data_dir: str, dataset_num: int, task_num: int, label_column, labelset, sample_size: int, full_label) \
        -> DatasetDict[str, pd.DataFrame]:
    datasets = DatasetDict()

    train_dataset_task_files = glob.glob(os.path.join(data_dir, f'ds_{dataset_num}__task_{task_num}_train_set*.csv'))
    eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set'
    eval_df = pd.read_csv(os.path.join(data_dir, eval_set_name + '.csv'))
    if dataset_num ==4:
        eval_df['text'] = eval_df['title_h1'] + ' ' + eval_df['text_200']
    eval_df[label_column] = eval_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
    assert(sorted(eval_df[label_column].unique()) == sorted(labelset), "Unknown values in the test data!")
    datasets["eval"] = Dataset.from_pandas(eval_df)

    if sample_size == 'all':
        train_dfs_ = {fn.strip('.csv'): pd.read_csv(fn) for fn in train_dataset_task_files}
        datasets.update(train_dfs_)
    else:
        train_df_fn = f'ds_{dataset_num}__task_{task_num}_train_set_{sample_size}'
        train_df = pd.read_csv(os.path.join(data_dir, train_df_fn + '.csv'))
        train_df[label_column] = train_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
        if dataset_num ==4:
            train_df['text'] = train_df['title_h1'] + ' ' + train_df['text_200']
        datasets["train"] = Dataset.from_pandas(train_df)
        if train_df_fn not in [os.path.basename(fn).strip('.csv') for fn in train_dataset_task_files]:
            raise ValueError(f"Sample size {sample_size} not found for"
                             f" dataset {dataset_num} and task {task_num}")

    return datasets

def load_full_dataset(data_dir: str, dataset_num: int, task_num: int, label_column, labelset, sample_size, full_label) \
        -> DatasetDict[str, pd.DataFrame]:
    datasets = DatasetDict()

    eval_set_name = f'ds_{dataset_num}__task_{task_num}_full__for_zero_shot_classification'
    eval_df = pd.read_csv(os.path.join(data_dir, eval_set_name + '.csv'))
    if dataset_num ==4:
        eval_df['text'] = eval_df['title_h1'] + ' ' + eval_df['text_200']

    eval_df[label_column] = eval_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
    assert(sorted(eval_df[label_column].unique()) == sorted(labelset), "Unknown values in the test data!")
    datasets["eval"] = Dataset.from_pandas(eval_df)
    #just to not break the trainer, not needed
    train_df_fn = f'ds_{dataset_num}__task_{task_num}_train_set_{sample_size}'
    train_df = pd.read_csv(os.path.join(data_dir, train_df_fn + '.csv'))
    train_df[label_column] = train_df[label_column].map(str)
    datasets["train"] = Dataset.from_pandas(train_df)

    return datasets


def _process_dataset(data_dir, dataset_num, task_num, dataset_name, label_column, labelset, full_label):
    df = pd.read_csv(os.path.join(data_dir, dataset_name + '.csv'))
    df[label_column] = (df[label_column]
                             .apply(lambda x: map_label_to_completion(x, task_num, full_label=full_label)))
    if dataset_num == 4:
        df['text'] = df['title_h1'] + ' ' + df['text_200']

    df = df[['text', label_column]]

    assert all([label in labelset for label in df[label_column].unique().tolist()]), \
        "Unknown values in the test data!"

    return df
