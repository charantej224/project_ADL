import os
from utils.json_utils import read_json
import pandas as pd

root_dir_path = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
prob_category = os.path.join(root_dir_path, "problem/prob_category.json")
dept_category = os.path.join(root_dir_path, "dept/dept_category.json")
prob_category = read_json(prob_category)
dept_category = read_json(dept_category)


def apply_dept(input_string):
    return list(dept_category.values()).index(input_string)


re_map = {
    "Parks & Rec": "Parks and Rec",
    "Parks & Recreation": "Parks and Rec",
    "Information Technology": "DROP",
    "NCS": "DROP",
    "Housing Community Dev": "DROP",
    "IT": "DROP",
    "Municipal Court": "DROP"
}


def apply_process_dept(input_string):
    if input_string in list(re_map.keys()):
        return re_map[input_string]
    else:
        return input_string


def process_df_prob(input_df):
    list_df = []
    for each in list(prob_category.values()):
        list_df.append(input_df[input_df['PROB_LABEL'] == each])
    final_df = pd.concat(list_df).reset_index()
    return final_df


def apply_prob(input_string):
    return list(prob_category.values()).index(input_string)
