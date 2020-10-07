from dataset.datahandler import load_datasets
from models.model_train_test import start_epochs, load_model
import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
classification = os.path.join(root_dir, "dept_classification.csv")
classification_df = pd.read_csv(classification)
classification_df.rename(columns={"DESCRIPTION": "desc", "DEPARTMENT_ID": "label"}, inplace=True)

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
model_directory = os.path.join(root_dir, "dept_state_dict")
metrics_json = os.path.join(root_dir, "dept_metric.json")
load_model_path = os.path.join(root_dir, 'model_state_dict2.pt')

training_loader, testing_loader = load_datasets(classification_df, train_size=0.8)
start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=3)
#load_model(load_model_path, training_loader, testing_loader)
