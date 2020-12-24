import os
import pandas as pd
from utils.json_utils import write_json, read_json
import matplotlib.pyplot as plt

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
final_data = "final_data/Data_with_no_desc.csv"
final_data = os.path.join(root_dir, final_data)


def process_final_data():
    data_311 = pd.read_csv(final_data)
    data_311 = data_311[data_311['CREATION YEAR'] >= 2015]
    list_records_dept = data_311.groupby(by=['CREATION YEAR', 'ZIP CODE', 'DEPARTMENT'])['CASE ID'].count().reset_index(
        name='COUNT').to_dict('records')
    list_records_category = data_311.groupby(by=['CREATION YEAR', 'ZIP CODE', 'CATEGORY'])[
        'CASE ID'].count().reset_index(
        name='COUNT').to_dict('records')

    write_json(list_records_dept, "year_dept.json")
    write_json(list_records_category, "year_category.json")


def merge_json():
    final_dict = {}
    dept_dict = read_json("year_dept.json")
    category_dict = read_json("year_category.json")
    for each in dept_dict:
        each["CREATION YEAR"] = str(each["CREATION YEAR"])
        each["ZIP CODE"] = str(int(each["ZIP CODE"]))

        if each["CREATION YEAR"] not in final_dict.keys():
            final_dict[each["CREATION YEAR"]] = {}
        if each["ZIP CODE"] not in final_dict[each["CREATION YEAR"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]] = {}
        if "DEPARTMENT" not in final_dict[each["CREATION YEAR"]][each["ZIP CODE"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["DEPARTMENT"] = []
        dept = {"name": each["DEPARTMENT"], "count": each["COUNT"]}
        final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["DEPARTMENT"].append(dept)

    for each in category_dict:
        each["CREATION YEAR"] = str(each["CREATION YEAR"])
        each["ZIP CODE"] = str(int(each["ZIP CODE"]))
        if each["CREATION YEAR"] not in final_dict.keys():
            final_dict[each["CREATION YEAR"]] = {}
        if each["ZIP CODE"] not in final_dict[each["CREATION YEAR"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]] = {}
        if "CATEGORY" not in final_dict[each["CREATION YEAR"]][each["ZIP CODE"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["CATEGORY"] = []
        dept = {"name": each["CATEGORY"], "count": each["COUNT"]}
        final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["CATEGORY"].append(dept)
    write_json(final_dict, "final.json")


def process_time_series_data():
    print("values.")
    data_311 = pd.read_csv(final_data)
    data_311 = data_311[data_311['CREATION YEAR'] >= 2015]
    data_311['DAYS TO CLOSE'] = data_311['DAYS TO CLOSE'].apply(lambda x: str(x).replace(",", ""))
    data_311['DAYS TO CLOSE'] = data_311['DAYS TO CLOSE'].astype("float64")
    time_series = data_311.groupby(by=['CREATION YEAR', 'CREATION MONTH', 'ZIP CODE', 'DEPARTMENT'])[
        'DAYS TO CLOSE'].mean().reset_index().to_dict('records')
    write_json(time_series, "time_series.json")


def process_json():
    time_series_dict = read_json("time_series.json")
    final_dict = {}
    for each in time_series_dict:
        each["CREATION YEAR"] = str(each["CREATION YEAR"])
        each["ZIP CODE"] = str(int(each["ZIP CODE"]))
        each["CREATION MONTH"] = str(each["CREATION MONTH"])

        if each["CREATION YEAR"] not in final_dict.keys():
            final_dict[each["CREATION YEAR"]] = {}
        if each["ZIP CODE"] not in final_dict[each["CREATION YEAR"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]] = {}
        if "DEPARTMENT" not in final_dict[each["CREATION YEAR"]][each["ZIP CODE"]].keys():
            final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["DEPARTMENT"] = []

        dept = {
            "CREATION MONTH": each["CREATION MONTH"],
            "NAME": each["DEPARTMENT"],
            "COUNT": each["DAYS TO CLOSE"]
        }
        final_dict[each["CREATION YEAR"]][each["ZIP CODE"]]["DEPARTMENT"].append(dept)
    write_json(final_dict, "time_series_final.json")


def apply_sequence(input_df):
    return 100 * input_df['CREATION YEAR'] + 10 * input_df['CREATION MONTH']


def process_ivanhoe():
    data_311 = pd.read_csv(final_data)
    data_311 = data_311[data_311["CREATION YEAR"] == 2019]
    data_311.fillna(value=0, inplace=True)
    data_311["ZIP CODE"] = data_311["ZIP CODE"].astype('int64')
    data_311 = data_311[data_311["DEPARTMENT"] == "NHS"]
    data_311['DAYS TO CLOSE'] = data_311['DAYS TO CLOSE'].apply(lambda x: str(x).replace(",", ""))
    data_311['DAYS TO CLOSE'] = data_311['DAYS TO CLOSE'].astype("float64")
    data_311 = data_311[data_311["DAYS TO CLOSE"] > 0]
    data_311['sequence'] = data_311.apply(apply_sequence, axis=1)

    data_311_zip1 = data_311[data_311["ZIP CODE"] == 64130]
    data_311_zip1 = data_311_zip1.sort_values(by=['sequence']).reset_index()
    data_311_zip1["MM/YYYY"] = data_311_zip1['CREATION MONTH'].astype('str') + "/" + data_311_zip1[
        'CREATION YEAR'].astype('str')
    data_311_zip1 = data_311_zip1[['sequence', 'MM/YYYY', 'DAYS TO CLOSE']]
    data_311_zip2 = data_311[data_311["ZIP CODE"] == 64110]
    data_311_zip2 = data_311_zip2.sort_values(by=['sequence']).reset_index()
    data_311_zip2["MM/YYYY"] = data_311_zip2['CREATION MONTH'].astype('str') + "/" + data_311_zip2[
        'CREATION YEAR'].astype('str')
    data_311_zip2 = data_311_zip2[['sequence', 'MM/YYYY', 'DAYS TO CLOSE']]
    data_311_zip1.drop_duplicates('sequence', inplace=True)
    data_311_zip1.rename(columns={'DAYS TO CLOSE': '64130'}, inplace=True)
    data_311_zip2.rename(columns={'DAYS TO CLOSE': '64110'}, inplace=True)
    data_311_zip2.drop_duplicates('sequence', inplace=True)
    merged = data_311_zip1.merge(data_311_zip2, on=['sequence'])
    merged.to_csv("merged.csv")
    # ax = data_311_zip1.plot()
    # data_311_zip2.plot(ax=ax)
    fig, ax = plt.subplots()
    ax.plot(merged['sequence'], merged['64130'], label="ZIP 64130")
    ax.plot(merged['sequence'], merged['64110'], label="ZIP 64110")
    ax.ticklabel_format(style='plain')
    ax.ticklabel_format(useOffset=False)
    ax.title('Time Series for NHS in zip codes')

    # plt.plot(data_311_zip2['DAYS TO CLOSE'], data_311_zip2['sequence'])
    plt.show()


if __name__ == '__main__':
    process_ivanhoe()
