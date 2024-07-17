import glob
import os
import re
import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score,
                             precision_score, f1_score, precision_recall_fscore_support)

###########################################################################
# UTILS SCRIPT 1- DATA LABELLING
# This is an auxiliary script to convert the coded values to match the labels provided.
# If your task does not correspond to one of the examples, add your mappings in place of all "INSERT YOUR TASK HERE" comments.
############################################################################



# Function 1 - Define correspondance between short and full labels
# This is auxiiliary for validation scripts to convert zero-shot output, which frequently does not return data in the format asked.
task_to_display_labels = {
    1: {
        'full_name': ['RELEVANT', 'IRRELEVANT'], 'short_name': ['A', 'B'],
    },

    2: {
        'full_name': ['PROBLEM', 'SOLUTION', 'BOTH','NEITHER'], 'short_name': ['A', 'B', 'C', 'D'],
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

    ### INSERT YOUR TASK HERE

}


# Function 2 - Map the original label (y_true) to the completion
# if your data is already in the format required, replace function with "return label"
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
                new_label = label.upper() #if label != 'Neither' else 'NEUTRAL'
                assert new_label in ['SOLUTION', 'PROBLEM', 'NEITHER', 'BOTH']
            else:
                new_label_mapping = {'Problem': 'A', 'Solution': 'B', 'Both': 'C', 'Neither': 'D'}
                new_label = new_label_mapping[label]
                assert new_label in ['A', 'B', 'C', 'D']

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

        #INSET YOUR TASK HERE

    except Exception as e:
        print(e)
        print("label not mapped")
        #new_label != '', f"Label {label} could not be mapped to a completion"

    return new_label

#NOTE: originally, there used to be a function here to convert the output of the model to a label needed. However, the output differs by model, so these functions have been moved to individual validation scripts.

# Function 3- Sklearn metrics for outcome evaluation
default_metrics = {
    'accuracy': accuracy_score,
    'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division="warn", average='macro'),
    'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division="warn", average='macro'),
    'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division= "warn", average ='macro')
}

#Function 4 - Metrics and Confusion Matrix Plotting
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

def plot_count_and_normalized_confusion_matrix_by_class(y_true, y_pred, display_labels, labels, xticks_rotation='horizontal',
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
      # Calculate metrics
    metrics_result = {}

    for metric_name, metric_func in metrics.items():
        metrics_result[metric_name] = metric_func(y_true, y_pred)

    # Calculate class-wise metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    class_metrics = {}
    for i, label in enumerate(labels):
        class_metrics[f'{label}_precision'] = precision[i]
        class_metrics[f'{label}_recall'] = recall[i]
        class_metrics[f'{label}_f1'] = f1[i]
    metrics_result.update(class_metrics)

    return fig, cls_report, metrics_result