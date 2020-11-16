import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
master_dataframe_file_pred = os.path.join(root_dir, "final_data/Data_with_no_desc.csv")
final_df = pd.read_csv(master_dataframe_file_pred)
print(final_df.shape)
