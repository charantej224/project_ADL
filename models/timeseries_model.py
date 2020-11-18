import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
time_series_data = os.path.join(root_dir, 'time-series/time_series.csv')
time_series_df = pd.read_csv(time_series_data)
time_series_df.dropna(inplace=True)
time_series_df["DEPARTMENT"] = time_series_df["DEPARTMENT"].astype("str")
time_series_df["PROB_LABEL"] = time_series_df["PROB_LABEL"].astype("str")
time_series_df.sort_values(by=['CASE ID'], inplace=True)
time_series_df.reset_index(inplace=True)

max_prediction_length = 6
max_encoder_length = 30

training = TimeSeriesDataSet(
    time_series_df,
    time_idx="index",
    target="DAYS TO CLOSE",
    group_ids=["DEPARTMENT", "PROB_LABEL"],
    min_encoder_length=4 // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["DEPARTMENT", "PROB_LABEL"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    add_relative_time_idx=True,
    add_encoder_length=True,
    allow_missings=True
)

print(time_series_df.shape)
