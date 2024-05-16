from datasets import Dataset, DatasetDict
import pandas as pd 
import os
import glob
from label_utils import map_label_to_completion

###########################################################################
# UTILS SCRIPT 1- DATA LOADING
# This utils scripts loads training and evaluation datasets. If your files are not labeled in the way we provide, you will need to change the names here.
# It is highly recommended to keep the data in the way we provided in the repository (ds_1_task_1_train_set and ds_1_task_1_eval_set), as otherwise you will need to change the names of training and evaluation sets many times in other scripts.
############################################################################


# Function 1 - Load train and evaluation datasets for finetuning
def load_train_and_eval_datasets(data_dir: str, eval_set_name: str, train_set_name: str, task_num: int, label_column, labelset, full_label) \
        -> DatasetDict[str, pd.DataFrame]:
    datasets = DatasetDict()

    eval_df = pd.read_csv(os.path.join(data_dir, eval_set_name + '.csv'))
    eval_df[label_column] = eval_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
    assert sorted(eval_df[label_column].unique()) == sorted(labelset), "Unknown values in the test data!"
    datasets["eval"] = Dataset.from_pandas(eval_df)

    # if sample_size == 'all':
    #     train_dfs_ = {fn.strip('.csv'): pd.read_csv(fn) for fn in train_dataset_task_files}
    #     datasets.update(train_dfs_)
    # else:
    train_df = pd.read_csv(os.path.join(data_dir, train_set_name+ '.csv'))
    train_df[label_column] = train_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
    datasets["train"] = Dataset.from_pandas(train_df)

    return datasets

# Function 2 - Load full dataset only for zero-shot prediction
def load_full_dataset(data_dir: str, dataset_name: str, task_num: int, label_column, labelset, full_label) \
        -> DatasetDict[str, pd.DataFrame]:
    datasets = DatasetDict()

    eval_df = pd.read_csv(os.path.join(data_dir, dataset_name + '.csv'))
    eval_df[label_column] = eval_df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label))
    assert sorted(eval_df[label_column].unique()) == sorted(labelset), "Unknown values in the test data!"
    datasets["eval"] = Dataset.from_pandas(eval_df)

    return datasets

#Function 3- Check that combination of task and dataset exists in the mapping file
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


#Function 4 - Pre-process data
def preprocess_function(tokenizer, prompt, df, label_column, max_length: int = 4096, padding: str | bool = False):
    # first check that all inputs are part of a labelset
    inputs = [prompt.format(text=text_i) for text_i in df["text"]]
    model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True)

    labels = tokenizer(
        text_target=df[label_column],
        padding=padding,
        max_length=max_length,
        truncation=True,
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


#Function 5- Print trainable parameters


