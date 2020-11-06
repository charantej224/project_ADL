import pandas as pd
from fbprophet import Prophet
import os
from utils.json_utils import read_json, write_json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"
final_df_path = os.path.join(root_dir, "final_data/311_Cases_master_with_desc_with_prediction.csv")
test_train_df = os.path.join(root_dir, "final_data/Data_with_no_desc.csv")
dept_category = read_json(os.path.join(root_dir, "dept/dept_category.json"))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


value_dict = {}
# Python
final_df = pd.read_csv(final_df_path)
test_train_df = pd.read_csv(test_train_df)
test_train_df = test_train_df[test_train_df['CREATION YEAR'] > 2015]
train_split = 80

final_df['DAYS TO CLOSE'].fillna(0, inplace=True)
print(final_df['CREATION DATE'].isna().sum())
print(final_df['DAYS TO CLOSE'].isna().sum())

test_train_df['DAYS TO CLOSE'] = test_train_df['DAYS TO CLOSE'].apply(lambda x: str(x).replace(",", ""))

list_of_dataframes = []

for each_dept in sorted(list(dept_category.values())):
    print(f' processing - {each_dept}')
    each_test_train = test_train_df[test_train_df.DEPARTMENT == each_dept].reset_index()
    each_dept_df = final_df[final_df.DEPARTMENT == each_dept].reset_index()

    test_time_train = each_test_train[['CREATION DATE', 'DAYS TO CLOSE']]
    each_df = each_dept_df[['CREATION DATE', 'DAYS TO CLOSE']]

    each_df.rename(columns={'CREATION DATE': 'ds', 'DAYS TO CLOSE': 'y'}, inplace=True)
    test_time_train.rename(columns={'CREATION DATE': 'ds', 'DAYS TO CLOSE': 'y'}, inplace=True)
    # test_time_train.y.apply(lambda x: str(x).replace(",", ""))
    test_time_train.y = test_time_train.y.astype('float64')
    test_time_train.y.fillna(0, inplace=True)
    train, test = train_test_split(test_time_train, test_size=0.2)
    m = Prophet()
    m.fit(train)
    forecast = m.predict(test)
    mae_value = mean_absolute_error(test['y'].values, forecast['yhat'].values)
    mape_error = mean_absolute_percentage_error(test['y'].values, forecast['yhat'].values)
    print(f'mean absolute error : {mae_value},MAPE {mape_error} , department {each_dept}')
    metric_dict = {'MAE': mae_value, 'MAPE': mape_error}
    value_dict[each_dept] = metric_dict
    fig1 = m.plot(forecast)
    fig1.savefig(each_dept + ".png")
    whole_result = m.predict(each_df)
    each_df['TIME_PRED'] = whole_result['yhat']
    each_df['CASE ID'] = each_dept_df['CASE ID']
    list_of_dataframes.append(each_df)

write_json(value_dict, "time_series_metrics.json")
final_pred = pd.concat(list_of_dataframes)
final_pred.to_csv("final_val.csv", header=True, index=False)
