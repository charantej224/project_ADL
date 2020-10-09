import os
import json

df_root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/problem_classification"
category_json = os.path.join(df_root_dir, "category.json")
new_category_json = os.path.join(df_root_dir, "new_category.json")

with open(category_json, 'r') as f:
    category_dict = json.load(f)
    f.close()
new_set = set()
for i in range(440):
    value = category_dict[str(i)]
    category = value.split("-")[0]
    category = category.split("/")[0].strip()
    new_set.add(category)

print(len(new_set))
new_classification_dict = {}

for each in new_set:
    new_classification_dict[each] = each

with open(new_category_json, 'w') as f:
    json.dump(new_classification_dict, f, indent=2)
    f.close()
