import json
import pandas as pd
import os

df_root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/problem_classification"
prob_classify = os.path.join(df_root_dir, "problem_classification.csv")
tweaked_category_json = os.path.join(df_root_dir, "tweaked_category.json")

problem_classify = pd.read_csv(prob_classify)

tweaked_category_dict = {}

counter = 0
for each in sorted(list(problem_classify['label'].unique())):
    tweaked_category_dict[counter] = each
    counter += 1

with open(tweaked_category_json, 'w') as f:
    json.dump(tweaked_category_dict, f, indent=2)

problem_classify['label'] = problem_classify['label'].apply(lambda x: list(tweaked_category_dict.values()).index(x))
print("processed")
problem_classify.to_csv(prob_classify, header=True, index=False)
