import torch
from tqdm.notebook import tqdm
import pandas as pd
import os
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import json
from transformers import BertForSequenceClassification

root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
classification = os.path.join(root_dir, "dept_classification.csv")
category_json = os.path.join(root_dir, "category.json")

with open(category_json, 'r') as f:
    label_dict = json.load(f)
    f.close()
df = pd.read_csv(classification)
