import os
import pandas as pd
import torch
from models.LSTM_Model import LSTM
import numpy as np
from utils.apply_functions import apply_dept, apply_prob, apply_process_dept, process_df_prob
from sklearn.model_selection import train_test_split

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
time_series_data = os.path.join(root_dir, 'time-series/time_series.csv')
time_series_df = pd.read_csv(time_series_data)
time_series_df.dropna(inplace=True)
time_series_df["DEPARTMENT"] = time_series_df["DEPARTMENT"].astype("str")
time_series_df["PROB_LABEL"] = time_series_df["PROB_LABEL"].astype("str")
time_series_df.sort_values(by=['CASE ID'], inplace=True)
time_series_df.reset_index(inplace=True)

learning_rate = 0.0001
batch_size = 32


def apply_date_data(input_string):
    input_string = input_string.split("/")
    total = 0
    for each in input_string:
        total += int(each)
    return total


time_series_df['sequence_id'] = time_series_df['CREATION DATE'].apply(apply_date_data)
time_series_df['DEPARTMENT'] = time_series_df['DEPARTMENT'].apply(apply_process_dept)
time_series_df = time_series_df[time_series_df['DEPARTMENT'] != 'DROP']
time_series_df['DEPARTMENT'] = time_series_df['DEPARTMENT'].apply(apply_dept)
time_series_df = process_df_prob(time_series_df)
time_series_df['PROB_LABEL'] = time_series_df['PROB_LABEL'].apply(apply_prob)
x_data, y_data = time_series_df[['sequence_id', 'DEPARTMENT', 'PROB_LABEL']], time_series_df['DAYS TO CLOSE']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = torch.tensor(x_train.values), torch.tensor(x_test.values), torch.tensor(y_train.values), torch.tensor(
    y_test.values)
print(time_series_df.shape)

model = LSTM(3, 6, batch_size=batch_size, output_dim=1, num_layers=6)
loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#####################
# Train model
#####################

num_epochs = 10
hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    # Forward pass
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimiser.step()
