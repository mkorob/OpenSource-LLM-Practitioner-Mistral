import re
import os
import glob
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from label_utils import map_label_to_completion


def load_train_and_eval_datasets(data_dir: str, eval_set_name: str, train_set_name: str, task_num: int, label_column: str,labelset: list[str], full_label: bool, sample_size: int,
                                 system_prompt: str, user_prompt_format: str,
                                 llama_2: bool = False) \
        -> DatasetDict[str, pd.DataFrame | Dataset]:
    datasets = DatasetDict()

    #eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set' if not zero_shot_full_dataset \
    #    else f'ds_{dataset_num}__task_{task_num}_full__for_zero_shot_classification'

    # Process evaluation set
    eval_df = _process_dataset(data_dir, task_num, eval_set_name, label_column, labelset, full_label,
                               system_prompt, user_prompt_format, llama_2=llama_2)
    datasets["eval"] = Dataset.from_pandas(eval_df)

    eval_df_wo_completion = _process_dataset(data_dir, task_num, eval_set_name, label_column,
                                             labelset, full_label, system_prompt, user_prompt_format,
                                             no_completion=True, llama_2=llama_2)
    datasets["eval_wo_completion"] = Dataset.from_pandas(eval_df_wo_completion)

    # process trainset
    train_df = _process_dataset(data_dir, dataset_num, task_num, train_set_name, label_column, labelset, full_label,
                                system_prompt, user_prompt_format, llama_2=llama_2)

    datasets["train"] = Dataset.from_pandas(train_df)

    return datasets


def load_full_dataset(data_dir: str, dataset_name: str, ttask_num: int, label_column: str,labelset: list[str], full_label: bool, sample_size: int, system_prompt: str, user_prompt_format: str,llama_2: bool = False) \
        -> DatasetDict[str, pd.DataFrame | Dataset]:
    datasets = DatasetDict()

    #eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set' if not zero_shot_full_dataset \
    #    else f'ds_{dataset_num}__task_{task_num}_full__for_zero_shot_classification'

    # Process evaluation set
    eval_df = _process_dataset(data_dir, task_num, dataset_name, label_column, labelset, full_label,
                               system_prompt, user_prompt_format, llama_2=llama_2)
    datasets["eval"] = Dataset.from_pandas(eval_df)

    eval_df_wo_completion = _process_dataset(data_dir, task_num, dataset_name, label_column,
                                             labelset, full_label, system_prompt, user_prompt_format,
                                             no_completion=True, llama_2=llama_2)
    datasets["eval_wo_completion"] = Dataset.from_pandas(eval_df_wo_completion)

    return datasets

def _process_dataset(data_dir, task_num, dataset_name, label_column, labelset, full_label,
                     system_prompt, user_prompt_format, no_completion=False, llama_2=False):
    print(f"loading {os.path.join(data_dir, dataset_name + '.csv')}") 
    print(os.path.join(data_dir, dataset_name + '.csv'))
    df = pd.read_csv(os.path.join(data_dir, dataset_name + '.csv'))
    print('dataset has the following cols', df.columns)
    print('The label_column is:', label_column) 
    df[label_column] = df[label_column].apply(lambda x: map_label_to_completion(x, task_num, full_label=full_label))

    labelset += [label.upper() for label in labelset]

    assert all([label in labelset for label in df[label_column].unique().tolist()]), \
        "Unknown values in the test data!"

    generate_prompt = generate_prompt_oasst if not llama_2 else generate_official_llama2_prompt

    df['text'] = df.apply(
        lambda row: generate_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt_format.format(text=row['text']),
            completion=row[label_column] if not no_completion else None),
        axis=1)

    df = df[['text']]

    return df

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


def generate_prompt_oasst(system_prompt: str, user_prompt: str, completion: Optional[str] = None, new_format: bool = False):
    # As we are using an older version of the model, we need to use the old format specified in:
    #  https://github.com/LAION-AI/Open-Assistant/blob/main/model/MESSAGE_AND_TOKEN_FORMAT.md#message-format-v2

    system_prompt_str = f"<|system|>{system_prompt}<|endoftext|>" if new_format else system_prompt
    user_prompt_str = f"<|prompter|>{user_prompt}<|endoftext|>"

    if completion is None:
        completion_str = "<|assistant|>"

    else:
        completion_str = f"<|assistant|>{completion}<|endoftext|>"

    prompt = system_prompt_str + user_prompt_str + completion_str

    # Remove break lines
    prompt = prompt.replace('\n', ' ')

    return prompt


def clean_text(text):
    cleaned_text = re.sub(r'\r', '', text)
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
    return cleaned_text


def generate_official_llama2_prompt(system_prompt, user_prompt, completion=None):
    user_prompt = clean_text(user_prompt)

    code_template = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    if completion is not None:
        completion = clean_text(completion)
        code_template += f" {completion} </s>"

    return code_template
