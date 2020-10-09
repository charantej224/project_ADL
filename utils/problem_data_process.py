import os
import pandas as pd
import json

df_root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/problem_classification"
source_root = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"
final_df = "merged_source.csv"
prob_classify = "problem_classification.csv"
prob_classify = os.path.join(df_root_dir, prob_classify)
category_json = os.path.join(df_root_dir, "category.json")
new_category_json = os.path.join(df_root_dir, "new_category.json")
final_df = os.path.join(source_root, final_df)

with open(new_category_json, 'r') as f:
    new_category_map = json.load(f)


def get_category_dict(input_df, column_name, category_json_name):
    counter = 0
    category_dict = {}
    for each in sorted(list(input_df[column_name].unique())):
        category_dict[counter] = each
        category_dict[each] = counter
        counter += 1
    with open(category_json_name, 'w') as f:
        json.dump(category_dict, f, indent=2)
        f.close()
    return category_dict


def apply_transform(input):
    category = input.split("-")[0]
    category = category.split("/")[0].strip()
    return new_category_map[category]


master_df = pd.read_csv(final_df)

category_dict = get_category_dict(master_df, 'REQUEST TYPE', category_json)
master_df['label'] = master_df['REQUEST TYPE'].apply(apply_transform)
problem_classify_df = master_df[['DESCRIPTION', 'label']]

problem_classify_df.rename(columns={"DESCRIPTION": "desc"}, inplace=True)
problem_classify_df.to_csv(prob_classify, header=True, index=False)

print("processed")
