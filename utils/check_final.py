import os
import pandas as pd

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project/"
generated_final_predictions = os.path.join(root_dir, "prediction_final.csv")
values = pd.read_csv(generated_final_predictions)
print("finished")
