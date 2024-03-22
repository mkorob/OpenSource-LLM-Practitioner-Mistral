import glob
import os
import re
import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score,
                             precision_score, f1_score)


task_num_to_task_name = {
    1: '1: Relevance',
    2: '2: Problem/Solution Frames',
    3: '3: Policy Frames',
    4: '4: Stance Detection',
    5: '5: Topics',
    6: '6: Policy SuperFrame'
}

dataset_num_to_dataset_name = {
    1: '1: Content Moderation Tweets',
    2: '2: Content Moderation Tweets 2023',
    3: '3: US Congress Members Tweets',
    4: '4: Content Moderation News Articles'
}

task_to_display_labels = {
    1: {
        'full_name': ['RELEVANT', 'IRRELEVANT'], 'short_name': ['A', 'B'],
    },

    2: {
        'full_name': ['PROBLEM', 'SOLUTION', 'NEUTRAL'], 'short_name': ['A', 'B', 'C'],
    },
    3: {
        'full_name': ['ECONOMY', 'MORALITY', 'FAIRNESS AND EQUALITY', 'POLICY PRESCRIPTION AND EVALUATION',
                      'LAW AND ORDER, CRIME AND JUSTICE', 'SECURITY AND DEFENSE', 'HEALTH AND SAFETY',
                      'QUALITY OF LIFE', 'POLITICAL', 'EXTERNAL REGULATION AND REPUTATION', 'OTHER'],
        'short_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    },
    4: {
        'full_name': ['IN FAVOR OF', 'AGAINST', 'NEUTRAL'], 'short_name': ['A', 'B', 'C'],
    },
    5: {
        'full_name': ['Section 230', 'Trump ban', 'Twitter Support', 'Platform Policies', 'Other', 'Complaint'],
        'short_name': ['A', 'B', 'C', 'D', 'E', 'F']
    },

    6:  {
        'full_name': ['policy and regulation', 'morality and law', 'economics', 'other'],
         'short_name': ['A', 'B', 'C', 'D']
    }

}

default_metrics = {
    'accuracy': accuracy_score,
    'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division="warn", average='macro'),
    'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division="warn", average='macro'),
    'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division= "warn", average ='macro')
}


def plot_count_and_normalized_confusion_matrix(y_true, y_pred, display_labels, labels, xticks_rotation='horizontal',
                                               metrics: dict = default_metrics):
    # Print classification report
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(cls_report)

    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Remove labels and display_labels not present in y_true
    display_labels = [label for label in display_labels if label in y_true.unique()]
    labels = [label for label in labels if label in y_true.unique()]

    # Plot count confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=display_labels)
    cm_disp.plot(ax=ax1, xticks_rotation=xticks_rotation)
    ax1.set_title('Count Confusion Matrix')

    # Plot normalized confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, display_labels=display_labels,
                                                      normalize='true')
    cm_disp.plot(ax=ax2, xticks_rotation=xticks_rotation)
    ax2.set_title('Normalized Confusion Matrix')

    # Show plot
    #plt.show()
    #plt.close()

    # Calculate metrics
    metrics = {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

    return fig, cls_report, metrics


def map_outputs_task_1(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|(a(\.|:|\)))|(\s|^|\')relev(a|e)nt', output.lower().strip()):
        return 'Relevant'
    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|not relevant|irrelevant|\s+b$', output.lower().strip()):
        return 'Irrelevant'
    elif output == np.nan or output == 'nan':
        return np.nan
    else:
        print(f'Weird value: {output.lower().strip()}')
        return np.nan


def map_outputs_task_2(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|(as a |a ){0,1}(pro|challeng)|\s+a$', output.lower().strip()):
        return 'Problem'
    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|(as a |a ){0,1}(solution|so)|\s+b$', output.lower().strip()):
        return 'Solution'
    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|neither|neutral|(\s)+c$', output.lower().strip()):
        return 'Neutral'
    elif output == np.nan or output == 'nan':
        return np.nan
    else:
        print(f'Weird value: {output.lower().strip()}')
        return np.nan


def map_outputs_task_3(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|economic|economy', output.lower().strip()):
        return 'economic'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|morality', output.lower().strip()):
        return 'morality'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|fairness and equality', output.lower().strip()):
        return 'fairness and equality'

    elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|policy prescription and evaluation|prescription and evaluation',
                   output.lower().strip()):
        return 'policy prescription and evaluation'

    elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|law and order|crime and justice|law enforcement', output.lower().strip()):
        return 'law and order, crime and justice'

    elif re.search(r'^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|security and defense', output.lower().strip()):
        return 'security and defense'

    elif re.search(r'^(answer:){0,1}(\s)*g(\s)*$|g(\.|:|\))|health and safety', output.lower().strip()):
        return 'health and safety'

    elif re.search(r'^(answer:){0,1}(\s)*h(\s)*$|h(\.|:|\))|quality of life', output.lower().strip()):
        return 'quality of life'

    elif re.search(r'^(answer:){0,1}(\s)*i(\s)*$|i(\.|:|\))|political', output.lower().strip()):
        return 'political'

    elif re.search(r'^(answer:){0,1}(\s)*j(\s)*$|j(\.|:|\))|external (regulation|region) and reputation|external regulation', output.lower().strip()):
        return 'external regulation and reputation'

    elif re.search(
            r'^(answer:){0,1}(\s)*k(\s)*$|(k|n|w)(\.|:|\))|other|climate change|leadership and executive responsibility|'
            r'expansion of service opportunities|access to higher ed|potential',
            output.lower().strip()):
        return 'other'

    elif output == np.nan or output == 'nan':
        return np.nan

    else:
        print(f'Weird value: {output.lower().strip()}')
        return np.nan


def map_outputs_task_4(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|positive|postive stance|in favor|in advantage of', output.lower().strip()):
        return 'Positive Stance'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|negative|negative stance|against|aggainst', output.lower().strip()):
        return 'Negative Stance'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|neutral|neutral stance', output.lower().strip()):
        return 'Neutral Stance'

    elif output == np.nan or output == 'nan':
        return np.nan

    else:
        print(f'Weird value: {output.lower().strip()}')
        return np.nan


def map_outputs_task_5(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|section 230', output.lower().strip()):
        return 'section 230'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|trump ban|ban donald trump|ban(ning){0,1} trump', output.lower().strip()):
        return 'trump ban'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|twitter support', output.lower().strip()):
        return 'twitter support'

    elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|platform policies', output.lower().strip()):
        return 'platform policies'

    elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|complaint(s)+', output.lower().strip()):
        return 'general/personal complaint'

    elif re.search('^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|other|censorship|banning free speech|'
                   'blocking political opinions|state bankruptcy|republique|problems with tech platform|'
                   'problem with the application',
                   output.lower().strip()):
        return 'other'

    elif output == np.nan or output == 'nan':
        return np.nan

    else:
        print(f'Weird value: {output.lower().strip()}')
        return 'other'
    
def map_outputs_task_6(output):
    if re.search(r'^(answer:){0,1}(\s)*a(\s)*$|a(\.|:|\))|economic|economy', output.lower().strip()):
        return 'economics'

    elif re.search(r'^(answer:){0,1}(\s)*b(\s)*$|b(\.|:|\))|morality', output.lower().strip()):
        return 'morality and law'

    elif re.search(r'^(answer:){0,1}(\s)*c(\s)*$|c(\.|:|\))|fairness and equality', output.lower().strip()):
        return 'morality and law'

    elif re.search(r'^(answer:){0,1}(\s)*d(\s)*$|d(\.|:|\))|policy prescription and evaluation|prescription and evaluation',
                   output.lower().strip()):
        return 'policy and regulation'

    elif re.search(r'^(answer:){0,1}(\s)*e(\s)*$|e(\.|:|\))|law and order|crime and justice|law enforcement', output.lower().strip()):
        return 'morality and law'

    elif re.search(r'^(answer:){0,1}(\s)*f(\s)*$|f(\.|:|\))|security and defense', output.lower().strip()):
        return 'economics'

    elif re.search(r'^(answer:){0,1}(\s)*g(\s)*$|g(\.|:|\))|health and safety', output.lower().strip()):
        return 'morality and law'

    elif re.search(r'^(answer:){0,1}(\s)*h(\s)*$|h(\.|:|\))|quality of life', output.lower().strip()):
        return 'economics'

    elif re.search(r'^(answer:){0,1}(\s)*i(\s)*$|i(\.|:|\))|political', output.lower().strip()):
        return 'policy and regulation'

    elif re.search(r'^(answer:){0,1}(\s)*j(\s)*$|j(\.|:|\))|external (regulation|region) and reputation|external regulation', output.lower().strip()):
        return 'policy and regulation'

    elif re.search(
            r'^(answer:){0,1}(\s)*k(\s)*$|(k|n|w)(\.|:|\))|other|climate change|leadership and executive responsibility|'
            r'expansion of service opportunities|access to higher ed|potential',
            output.lower().strip()):
        return 'other'

    elif output == np.nan or output == 'nan':
        return np.nan

    else:
        print(f'Weird value: {output.lower().strip()}')
        return np.nan


def get_accuracy_accross_tasks(output_dir):
    results_fps = sorted([fp for fp in os.listdir(output_dir) if fp.endswith('csv') and
                          not fp.endswith('_with_mapped_outputs.csv')])

    results = []
    for results_fp in results_fps:
        df = pd.read_csv(os.path.join(output_dir, results_fp))
        dataset_num = int(results_fp.split('__')[0].split('_')[-1])
        task_num = int(results_fp.split('__')[1].split('_')[-1])
        model_name = results_fp.split('__')[2]

        # Filter out any rows without status_id (i.e: not valid rows)
        df = df[df['status_id' if dataset_num != 4 else 'id'].notna()]
        total_rows = df.shape[0]

        # Check if any outputs from the model reached OOM
        oom_df = df[df['model_output'].astype(str) == 'OOM']
        if oom_df.shape[0] > 0:
            os.makedirs(os.path.join(output_dir, 'OOM'), exist_ok=True)
            print('#'* 50)
            print(f'Model {model_name} on dataset {dataset_num}, task {task_num} reached OOM on {oom_df.shape[0]} rows')
            print('#' * 50)
            oom_df.to_csv(os.path.join(output_dir, 'OOM', results_fp), index=False)

            if oom_df.shape[0] == df.shape[0]:
                print('All rows were OOM')
            else:
                print('Around {:.2f}% of rows were OOM'.format(oom_df.shape[0] / df.shape[0] * 100))

            print('skipping this dataset and task\n\n' + '#'* 50 + '\n\n')
            continue

        x_tick_rotation = 'horizontal'

        try:
            if task_num == 1:
                df[f'relevant_flan_{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_1)
                y_true = df.loc[df['relevant_ra'].notna().index, 'relevant_ra']\
                    .map(lambda num: 'Relevant' if num == 1 else 'Irrelevant')
                y_pred = df.loc[y_true.index, f'relevant_flan_{model_name}']
                display_labels = ['Relevant', 'Irrelevant']
                labels = display_labels

            elif task_num == 2:
                df[f'problem_solution_{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_2)
                y_true = df['problem_solution_ra']
                y_pred = df[f'problem_solution_{model_name}']
                display_labels = ['Problem', 'Solution', 'Neutral']
                labels = display_labels

            elif task_num == 3:
                df[f'frames_{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_3)

                y_true = df['frame'] if dataset_num == 1 else df['frames_ra']
                allowed_frames = ['economic', 'morality', 'fairness and equality', 'policy prescription and evaluation',
                                  'law and order, crime and justice', 'security and defense',
                                  'health and safety', 'quality of life', 'political',
                                  'external regulation and reputation', 'other']
                y_true = y_true.map(lambda frame: 'other' if frame not in allowed_frames and isinstance(frame,
                                                                                                        str) and frame != 'nan' else frame)

                y_pred = df[f'frames_{model_name}']

                display_labels = allowed_frames
                labels = display_labels
                x_tick_rotation = 'vertical'

            elif task_num == 4:
                df[f'stance_ra__{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_4)
                y_true = df['stance_ra']
                y_pred = df[f'stance_ra__{model_name}']
                display_labels = ['Positive Stance', 'Negative Stance', 'Neutral Stance']
                labels = display_labels

            elif task_num == 5:
                df[f'topic'].fillna('other', inplace=True)
                df[f'topic'] = df[f'topic'].astype(str).map(
                    lambda topic: 'general/personal complaint' if topic in ['general complaint',
                                                                            'personal complaint'] else topic)
                df[f'topic_{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_5)

                y_true = df[f'topic']
                allowed_topics = ['section 230', 'trump ban', 'twitter support', 'platform policies',
                                  'general/personal complaint', 'other']
                y_true = y_true.map(lambda topic: 'other' if topic not in allowed_topics and isinstance(topic,
                                                                                                        str) and topic != 'nan' else topic)

                y_pred = df[f'topic_{model_name}']
                display_labels = allowed_topics
                labels = display_labels
            
            elif task_num == 6:
                df[f'frames_{model_name}'] = df['model_output'].astype(str).map(map_outputs_task_6)

                y_true = df['frame'] if dataset_num == 1 else df['frames_ra']
                allowed_frames = ['policy and regulation', 'morality and law', 'economics', 'other']
                y_true = y_true.map(lambda frame: 'other' if frame not in allowed_frames and isinstance(frame,
                                                                                                        str) and frame != 'nan' else frame)
                y_pred = df[f'frames_{model_name}']

                display_labels = allowed_frames
                labels = display_labels
                x_tick_rotation = 'vertical'


            else:
                raise ValueError(f'Unknown task number: {task_num}')

            # Save the results with the mapped model output
            os.makedirs(os.path.join(output_dir, 'mapped_outputs'), exist_ok=True)
            assert df.shape[0] == total_rows
            df.to_csv(os.path.join(output_dir, 'mapped_outputs', results_fp), index=False)

            # Choose valid indexes to use for evaluation
            y_true.dropna(inplace=True)
            total_labeled_rows = y_true.shape[0]
            num_rows_instruction_followed = y_pred[y_true.index][(y_pred.loc[y_true.index].notna()) &
                                                   (y_pred.loc[y_true.index] != 'nan')].shape[0]
            y_pred = y_pred.loc[y_true.index].fillna('NA')

            print('====================')
            print(results_fp)
            print('Dataset:', results_fp.split('__')[0].split('_')[-1])
            print('Model:', model_name)
            print('Task:', task_num)
            print('Accuracy:', accuracy_score(y_true, y_pred))
            print('Percentage of ground truth labeled rows: ', total_labeled_rows / total_rows)
            print('Percentage of rows instruction correctly followed:', num_rows_instruction_followed / total_labeled_rows)

            plot_count_and_normalized_confusion_matrix(y_true, y_pred, display_labels, labels, x_tick_rotation)

            print('====================')
            print('\n\n')

            results.append(
                {
                    'dataset': results_fp.split('__')[0].split('_')[-1],
                    'model': model_name,
                    'task': task_num,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'percentage_of_rows_used': total_labeled_rows / total_rows,
                    'percentage_of_rows_instruction_followed': num_rows_instruction_followed / total_labeled_rows
                }
            )

        except Exception as e:
            print('Problem with:', results_fp)
            print('e:', e)
            print('\n\n')

    results_df = pd.DataFrame(results)
    results_df.set_index(['task', 'dataset', 'model'], inplace=True)
    results_df.sort_index(inplace=True)

    return results_df

def load_full_dataset(data_dir: str, dataset_num: int, task_num: int) \
        -> dict[str, pd.DataFrame]:
    datasets = dict()
    eval_set_name = f'ds_{dataset_num}__task_{task_num}_eval_set_full'
    datasets[eval_set_name] = pd.read_csv(os.path.join(data_dir, f'ds_{dataset_num}__task_{task_num}_full__for_zero_shot_classification.csv'))

    if dataset_num == 4:
        for df in datasets.values():
            df['text'] = df['title_h1'] + ' ' + df['text_200']

    return datasets


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


def map_label_to_completion(label: str, task_num: int, full_label: bool = True) -> str:
    new_label = ''
    try:

        if task_num == 1:
            if full_label:
                if str(label) in ["0.0", "1.0", "1", "0"]:
                    new_label = 'RELEVANT' if int(label) == 1.0 or str(label) in ['1.0', '1'] else 'IRRELEVANT'
                else:
                    new_label= label.upper()
                assert new_label in ['RELEVANT', 'IRRELEVANT']
            else:
                if str(label) in ["0.0", "1.0", "1", "0"]:
                    new_label = 'A' if int(label) == 1.0 or str(label) in ['1.0', '1'] else 'B'
                else:
                    new_label= label.upper()
                assert new_label in ['A', 'B']

        elif task_num == 2:
            if full_label:
                new_label = label.upper() if label != 'Neither' else 'NEUTRAL'
                assert new_label in ['SOLUTION', 'PROBLEM', 'NEUTRAL']
            else:
                new_label_mapping = {'Problem': 'A', 'Solution': 'B', 'Neutral': 'C', 'Neither': 'C'}
                new_label = new_label_mapping[label]
                assert new_label in ['A', 'B', 'C']

        elif task_num == 3:
            if full_label:
                new_label_mapping = {
                    'economic': 'ECONOMY', 'economy': 'ECONOMY', 'morality': 'MORALITY', 'fairness and equality': 'FAIRNESS AND EQUALITY',
                    'policy prescription and evaluation': 'POLICY PRESCRIPTION AND EVALUATION',
                    'law and order, crime and justice': 'LAW AND ORDER, CRIME AND JUSTICE',
                    'security and defense': 'SECURITY AND DEFENSE', 'health and safety': 'HEALTH AND SAFETY',
                    'quality of life': 'QUALITY OF LIFE', 'political': 'POLITICAL',
                    'external regulation and reputation': 'EXTERNAL REGULATION AND REPUTATION', 'other': 'OTHER',
                    'capacity and resources': 'OTHER', 'public opinion': 'OTHER', 'cultural identity': 'OTHER',
                    'constitutionality and jurisprudence': 'OTHER'
                }
                new_label = new_label_mapping[label.lower()]
                assert new_label in ['ECONOMY', 'MORALITY', 'FAIRNESS AND EQUALITY', 'POLICY PRESCRIPTION AND EVALUATION',
                                    'LAW AND ORDER, CRIME AND JUSTICE', 'SECURITY AND DEFENSE', 'HEALTH AND SAFETY',
                                    'QUALITY OF LIFE', 'POLITICAL', 'EXTERNAL REGULATION AND REPUTATION', 'OTHER']
            else:
                new_label_mapping = {
                    'economic': 'A', 'economy': 'A','morality': 'B', 'fairness and equality': 'C',
                    'policy prescription and evaluation': 'D', 'law and order, crime and justice': 'E',
                    'security and defense': 'F', 'health and safety': 'G', 'quality of life': 'H',
                    'political': 'I', 'external regulation and reputation': 'J', 'other': 'K',
                    'capacity and resources': 'K', 'public opinion': 'K', 'cultural identity': 'K',
                    'constitutionality and jurisprudence': 'K'
                }
                new_label = new_label_mapping[label.lower()]
                assert new_label in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

        elif task_num == 4:
            if full_label:
                new_label_mapping = {
                    'Positive Stance': 'IN FAVOR OF', 'Negative Stance': 'AGAINST', 'Neutral Stance': 'NEUTRAL',
                    'IN FAVOR OF': 'IN FAVOR OF', 'AGAINST': 'AGAINST', 'NEUTRAL': 'NEUTRAL'
                }
                new_label = new_label_mapping[label]
                assert new_label in ['IN FAVOR OF', 'AGAINST', 'NEUTRAL']
            else:
                new_label_mapping = {
                    'Positive Stance': 'A', 'Negative Stance': 'B', 'Neutral Stance': 'C'
                }
                new_label = new_label_mapping[label]
                assert new_label in ['A', 'B', 'C']

        elif task_num == 5:
            if full_label:
                new_label_mapping = {
                    'section 230': 'SECTION 230', 'trump ban': 'TRUMP BAN', 'twitter support': 'TWITTER SUPPORT',
                    'platform policies': 'PLATFORM POLICIES', 'other': 'OTHER', 'general complaint': 'COMPLAINT',
                    'complaint': 'COMPLAINT',
                    'personal complaint': 'COMPLAINT'}
                new_label = new_label_mapping[str(label).lower()]
                assert new_label in ['SECTION 230', 'TRUMP BAN', 'TWITTER SUPPORT', 'PLATFORM POLICIES', 'OTHER',
                                    'COMPLAINT']
            else:
                new_label_mapping = {
                    'section 230': 'A', 'trump ban': 'B', 'twitter support': 'C', 'platform policies': 'D',
                    'general complaint': 'E', 'personal complaint': 'E', 'other': 'F',
                }
                new_label = new_label_mapping[label.lower()]
                assert new_label in ['A', 'B', 'C', 'D', 'E', 'F']

        elif task_num == 6:
            if full_label:
                new_label_mapping = {
                    'policy and regulation': 'POLICY AND REGULATION',
                    'morality and law': 'MORALITY AND LAW',
                    'economics': 'ECONOMICS',
                    'other': 'OTHER'
                }
                new_label = new_label_mapping[label.lower()]
                assert new_label in [ 'POLICY AND REGULATION', 'MORALITY AND LAW', 'ECONOMICS','OTHER']
            else:
                new_label_mapping = {
                    'policy and regulation': 'A',
                    'morality and law': 'B',
                    'economics': 'C',
                    'other': 'D'
                }
                new_label = new_label_mapping[label.lower()]
                assert new_label in ['A', 'B', 'C', 'D']

    except Exception as e:
        print(e)
        print("label not mapped")
        #new_label != '', f"Label {label} could not be mapped to a completion"

    return new_label
