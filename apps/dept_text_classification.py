from dataset.datahandler import load_datasets, load_test_datasets
from models.model_train_test import start_epochs, load_model
import os
import pandas as pd
import numpy as np

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"


def run_dept_bert_model():
    classification = os.path.join(root_dir, "dept/dept_classification_df.csv")
    classification_df = pd.read_csv(classification)
    number_of_classes = len(list(classification_df['label'].unique()))
    classification_df.rename(columns={"DESCRIPTION": "desc", "DEPARTMENT_ID": "label"}, inplace=True)
    model_directory = os.path.join(root_dir, "dept/dept_state_dict")
    metrics_json = os.path.join(root_dir, "dept/dept_metric.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10)
    load_model_path = os.path.join(root_dir, 'dept/dept_state_dict_9.pt')
    dept_test_loader = load_test_datasets(classification_df, number_of_classes)
    unique_ids, predictions = load_model(load_model_path, dept_test_loader, number_of_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'DEPT_LABEL'])
    dept_df.to_csv(os.path.join(root_dir, "dept/dept_df.csv"), index=False, header=True)


def run_problem_bert_model():
    classification = os.path.join(root_dir, "problem/prob_classification_df.csv")
    classification_df = pd.read_csv(classification)
    number_of_classes = len(list(classification_df['label'].unique()))
    print(sorted(list(classification_df['label'].unique())))
    model_directory = os.path.join(root_dir, "problem/prob_state_dict")
    metrics_json = os.path.join(root_dir, "problem/prob_metric.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10,
                 number_of_classes=number_of_classes)
    load_model_path = os.path.join(root_dir, 'problem/prob_state_dict_9.pt')
    prob_test_loader = load_test_datasets(classification_df, number_of_classes)
    unique_ids, predictions = load_model(load_model_path, prob_test_loader, number_of_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'PROB_LABEL'])
    dept_df.to_csv(os.path.join(root_dir, "problem/prob_df.csv"), index=False, header=True)


if __name__ == '__main__':
    run_problem_bert_model()
    run_dept_bert_model()
