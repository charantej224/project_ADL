from dataset.datahandler import load_datasets
from models.model_train_test import start_epochs, load_model
import os
import pandas as pd


def run_dept_bert_model():
    root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/dept_classification"
    classification = os.path.join(root_dir, "dept_classification.csv")
    classification_df = pd.read_csv(classification)
    classification_df.rename(columns={"DESCRIPTION": "desc", "DEPARTMENT_ID": "label"}, inplace=True)
    model_directory = os.path.join(root_dir, "dept_state_dict")
    metrics_json = os.path.join(root_dir, "dept_metric.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10)
    # load_model_path = os.path.join(root_dir, 'model_state_dict2.pt')
    # load_model(load_model_path, training_loader, testing_loader)


def run_problem_bert_model():
    root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/problem_classification"
    classification = os.path.join(root_dir, "problem_classification.csv")
    classification_df = pd.read_csv(classification)
    number_of_classes = len(list(classification_df['label'].unique()))
    print(sorted(list(classification_df['label'].unique())))
    model_directory = os.path.join(root_dir, "prob_state_dict")
    metrics_json = os.path.join(root_dir, "prob_metric.json")
    training_loader, testing_loader = load_datasets(classification_df, train_size=0.8,
                                                    number_of_classes=number_of_classes)
    start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=10,
                 number_of_classes=number_of_classes)
    # load_model_path = os.path.join(root_dir, 'prob_state_dict2.pt')
    # load_model(load_model_path, training_loader, testing_loader)


if __name__ == '__main__':
    run_problem_bert_model()
    run_dept_bert_model()
