import pandas as pd
import os
from utils.json_utils import read_json

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
problem_file = os.path.join(root_dir, "problem/prob_df.csv")
dept_file = os.path.join(root_dir, "dept/dept_df.csv")
merged_file = os.path.join(root_dir, "merged_prediction.csv")
dept_category_file = os.path.join(root_dir, "dept/dept_category.json")
prob_category_file = os.path.join(root_dir, "problem/prob_category.json")
master_dataframe_file = os.path.join(root_dir, "final_data/311_Cases_master_with_desc.csv")
master_dataframe_file_pred = os.path.join(root_dir, "final_data/311_Cases_master_with_desc_with_prediction.csv")


def merge_predictions():
    prob_df = pd.read_csv(problem_file)
    dept_df = pd.read_csv(dept_file)

    print(prob_df.shape)
    print(dept_df.shape)

    print(prob_df["PROB_LABEL"].isna().sum())
    print(dept_df["DEPT_LABEL"].isna().sum())
    merged_prediction = prob_df.merge(dept_df, how='inner', on="CASE ID")
    merged_prediction['CASE ID'] = merged_prediction['CASE ID'].astype(dtype='int64')
    merged_prediction['PROB_LABEL'] = merged_prediction['PROB_LABEL'].astype(dtype='int64')
    merged_prediction['DEPT_LABEL'] = merged_prediction['DEPT_LABEL'].astype(dtype='int64')
    merged_prediction.to_csv(merged_file, header=True, index=False)


def analyze_predictions():
    dept_category = read_json(dept_category_file)
    prob_category = read_json(prob_category_file)
    merged_df = pd.read_csv(merged_file)
    merged_df['DEPT_LABEL'] = merged_df['DEPT_LABEL'].apply(lambda x: dept_category[str(x)])
    merged_df['PROB_LABEL'] = merged_df['PROB_LABEL'].apply(lambda x: prob_category[str(x)])
    master_df = pd.read_csv(master_dataframe_file)
    new_master = master_df.merge(merged_df, on='CASE ID')
    new_master.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
    new_master.to_csv(master_dataframe_file_pred, header=True, index=False)


if __name__ == '__main__':
    merge_predictions()
    analyze_predictions()
