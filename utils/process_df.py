import pandas as pd
import os
from utils.json_utils import write_json, read_json

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
prob_file = os.path.join(root_dir, "problem/prob_classification_df.csv")
dept_file = os.path.join(root_dir, "dept/dept_classification_df.csv")
final_df = os.path.join(root_dir, "final_data/311_Cases_master_with_desc.csv")
dept_json = os.path.join(root_dir, "dept/dept_category.json")
prob_json = os.path.join(root_dir, "problem/prob_category.json")
map_prob_json = os.path.join(root_dir, "problem/new_category.json")  ## Used for mapping to parent category.
map_prob_json = read_json(map_prob_json)
master_dataframe_file_pred = os.path.join(root_dir, "final_data/Data_with_no_desc.csv")
timeseries_data = os.path.join(root_dir, 'time-series/time_series.csv')

final_data = pd.read_csv(final_df)

prob_category = {}
dept_category = {}


def apply_dept_label(input_text):
    return list(dept_category.values()).index(input_text)


def apply_prob_label(input_string):
    return list(prob_category.values()).index(input_string)


def apply_prob(input_text):
    category = input_text.split("-")[0]
    category = category.split("/")[0].strip()
    # category = map_prob_json[category]
    return category


def write_to_csv(df_to_write, file_name):
    df_to_write.to_csv(file_name, header=True, index=False)


def process_dept_df():
    counter = 0
    dept_df = final_data[['CASE ID', 'DESCRIPTION', 'DEPARTMENT']]
    for each in list(dept_df.DEPARTMENT.unique()):
        dept_category[str(counter)] = each
        counter += 1
    write_json(dept_category, dept_json)
    print('finished')
    dept_df['label'] = dept_df.DEPARTMENT.apply(apply_dept_label)
    dept_df.rename(columns={'CASE ID': 'u_id', 'DESCRIPTION': 'desc'}, inplace=True)
    write_to_csv(dept_df, dept_file)


def process_prob_df():
    counter = 0
    print(list(final_data['REQUEST TYPE'].unique()))
    final_data['label'] = final_data['REQUEST TYPE'].apply(apply_prob)
    print(list(final_data.label.unique()))
    for each in list(final_data.label.unique()):
        prob_category[str(counter)] = each
        counter += 1
    write_json(prob_category, prob_json)
    final_data['label'] = final_data['label'].apply(apply_prob_label)
    prob_df = final_data[['CASE ID', 'DESCRIPTION', 'label']]
    prob_df.rename(columns={'CASE ID': 'u_id', 'DESCRIPTION': 'desc'}, inplace=True)
    write_to_csv(prob_df, prob_file)
    print('finished')


def process_timeseries():
    actual_df = pd.read_csv(master_dataframe_file_pred)
    actual_df['PROB_LABEL'] = actual_df['REQUEST TYPE'].apply(apply_prob)
    timeseries_df = actual_df[['CASE ID', 'CREATION DATE', 'DEPARTMENT', 'PROB_LABEL', 'DAYS TO CLOSE']]
    timeseries_df.to_csv(timeseries_data, header=True, index=False)


if __name__ == '__main__':
    process_timeseries()
