import pandas as pd
import os
import json

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
final_output = "final_df_description.csv"
source_file = "Source_09_29_2020.csv"
merged_file = os.path.join(root_dir, "merged_source.csv")
category_json = os.path.join(root_dir, "category.json")
classification = os.path.join(root_dir, "dept_classification.csv")


def merge_dataframe(output_file):
    dataframe_list = []
    for each in os.listdir(root_dir):
        each_file = os.path.join(root_dir, each)
        dataframe_list.append(pd.read_csv(each_file))

    merged_pdf = pd.concat(dataframe_list)
    merged_pdf.to_csv(output_file, header=True, index=False)


def read_df_covert_dict(source_file_val):
    source_df = pd.read_csv(source_file_val)
    dict_vals = source_df.set_index("CASE ID").T.to_dict()
    return dict_vals


def write_json(final_df):
    counter = 0
    category_dict = {}
    for each in sorted(list(final_df['DEPARTMENT'].unique())):
        category_dict[counter] = each
        counter += 1
    with open(category_json, 'w') as f:
        json.dump(category_dict, f, indent=2)
        f.close()
    return category_dict


if __name__ == '__main__':
    final_output = os.path.join(root_dir, final_output)
    source_file = os.path.join(root_dir, source_file)

    # merge_dataframe(final_output)
    merged_df = pd.read_csv(final_output)
    merged_df = merged_df[["CASE ID", "DESCRIPTION"]]
    source_df = pd.read_csv(source_file)
    final_df = pd.merge(source_df, merged_df, on="CASE ID")
    final_df.to_csv(merged_file, header=True, index=False)
    category_map = write_json(final_df)
    # dict_vals = read_df_covert_dict(source_file)
    final_df['DEPARTMENT_ID'] = final_df['DEPARTMENT'].apply(lambda x: list(category_map.values()).index(x))
    classification_df = final_df[['DESCRIPTION', 'DEPARTMENT_ID']]
    classification_df.to_csv(classification, header=True, index=False)
    print("read Dataframe")
