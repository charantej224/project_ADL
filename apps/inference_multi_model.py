import os
import pandas as pd
import json
from dataset.datahandler import load_test_datasets
from models.model_train_test import load_model
import numpy as np

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"
dept_category = os.path.join(root_dir, "dept_classification/category.json")
prob_category = os.path.join(root_dir, "problem_classification/Problem_category.json")
prob_map_category = os.path.join(root_dir, "problem_classification/new_category.json")
generated_final_predictions = os.path.join(root_dir, "prediction_final.csv")


def get_catgory(input_file):
    with open(input_file, 'r') as f:
        category_dict = json.load(f)
        f.close()
        return category_dict


dept_category = get_catgory(dept_category)
prob_category = get_catgory(prob_category)
prob_map_category = get_catgory(prob_map_category)


def apply_dept(input_text):
    return list(dept_category.values()).index(input_text)


def apply_prob(input_text):
    category = input_text.split("-")[0]
    category = category.split("/")[0].strip()
    category = prob_map_category[category]
    return list(prob_category.values()).index(category)


def new_inference():
    final = os.path.join(root_dir, "final_data/311_Cases_master_with_desc.csv")
    final_data = pd.read_csv(final)
    department_df = final_data[['CASE ID', 'DESCRIPTION', 'DEPARTMENT']]
    problem_df = final_data[['CASE ID', 'DESCRIPTION', 'REQUEST TYPE']]
    department_df['label'] = department_df['DEPARTMENT'].apply(apply_dept)
    problem_df['label'] = problem_df['REQUEST TYPE'].apply(apply_prob)
    department_df.rename(columns={"CASE ID": "u_id", "DESCRIPTION": "desc"}, inplace=True)
    problem_df.rename(columns={"CASE ID": "u_id", "DESCRIPTION": "desc"}, inplace=True)
    department_df.drop(columns=['DEPARTMENT'], inplace=True)
    problem_df.drop(columns=['REQUEST TYPE'], inplace=True)
    dept_classes = len(list(pd.read_csv(department_df)['label'].unique()))
    prob_classes = len(list(pd.read_csv(problem_df)['label'].unique()))
    dept_test_loader = load_test_datasets(department_df, dept_classes)
    prob_test_loader = load_test_datasets(problem_df, prob_classes)
    load_dept_model_path = os.path.join(root_dir, 'dept_classification/dept_state_dict_9.pt')
    load_prob_model_path = os.path.join(root_dir, 'problem_classification/prob_state_dict_9.pt')
    unique_ids, predictions = load_model(load_dept_model_path, dept_test_loader, dept_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'DEPT_LABEL'])
    dept_df.to_csv(os.path.join(root_dir, "dept_df.csv"), index=False, header=True)
    unique_ids, predictions = load_model(load_prob_model_path, prob_test_loader, prob_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    prob_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'PROB_LABEL'])
    prob_df.to_csv(os.path.join(root_dir, "prob_df.csv"), index=False, header=True)


def inference_run():
    root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"
    dept_classification = os.path.join(root_dir, "dept_classification/dept_classification.csv")
    prob_classification = os.path.join(root_dir, "problem_classification/problem_classification.csv")
    merged_df = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/merged_source.csv"
    dept_classes = len(list(pd.read_csv(dept_classification)['DEPARTMENT_ID'].unique()))
    prob_classes = len(list(pd.read_csv(prob_classification)['label'].unique()))
    merged_data = pd.read_csv(merged_df)
    merged_data.drop_duplicates(inplace=True)
    department_df = merged_data[['CASE ID', 'DESCRIPTION', 'DEPARTMENT']]
    problem_df = merged_data[['CASE ID', 'DESCRIPTION', 'REQUEST TYPE']]
    department_df['label'] = department_df['DEPARTMENT'].apply(apply_dept)
    problem_df['label'] = problem_df['REQUEST TYPE'].apply(apply_prob)
    department_df.rename(columns={"CASE ID": "u_id", "DESCRIPTION": "desc"}, inplace=True)
    problem_df.rename(columns={"CASE ID": "u_id", "DESCRIPTION": "desc"}, inplace=True)
    department_df.drop(columns=['DEPARTMENT'], inplace=True)
    problem_df.drop(columns=['REQUEST TYPE'], inplace=True)
    dept_test_loader = load_test_datasets(department_df, dept_classes)
    prob_test_loader = load_test_datasets(problem_df, prob_classes)
    print("success")
    load_dept_model_path = os.path.join(root_dir, 'dept_classification/dept_state_dict_9.pt')
    load_prob_model_path = os.path.join(root_dir, 'problem_classification/prob_state_dict_9.pt')
    unique_ids, predictions = load_model(load_dept_model_path, dept_test_loader, dept_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    dept_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'DEPT_LABEL'])
    dept_df.to_csv(os.path.join(root_dir, "dept_df.csv"), index=False, header=True)
    unique_ids, predictions = load_model(load_prob_model_path, prob_test_loader, prob_classes)
    out_numpy = np.concatenate((unique_ids.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
    prob_df = pd.DataFrame(out_numpy, columns=['CASE ID', 'PROB_LABEL'])
    prob_df.to_csv(os.path.join(root_dir, "prob_df.csv"), index=False, header=True)
    prob_df["CASE ID"] = prob_df["CASE ID"].astype(np.int64)
    dept_df["CASE ID"] = dept_df["CASE ID"].astype(np.int64)
    final_df = merged_data.merge(prob_df, on='CASE ID').merge(dept_df, on='CASE ID')
    final_df['PROB_LABEL'] = final_df['PROB_LABEL'].apply(lambda x: str(x).replace(".0", ""))
    final_df['PROB_LABEL'] = final_df['PROB_LABEL'].apply(lambda x: prob_category[x])
    final_df['DEPT_LABEL'] = final_df['DEPT_LABEL'].apply(lambda x: str(x).replace(".0", ""))
    final_df['DEPT_LABEL'] = final_df['DEPT_LABEL'].apply(lambda x: dept_category[x])
    final_df.to_csv(generated_final_predictions, index=False, header=True)

    # load_model_path = os.path.join(dept_root_dir, 'dept_state_dict_9.pt')
    # load_model(load_model_path, training_loader, testing_loader)
    #
    # load_model_path = os.path.join(prob_root_dir, 'prob_state_dict_9.pt')
    # load_model(load_model_path, training_loader, testing_loader)


if __name__ == '__main__':
    new_inference()
