import pandas as pd
from dataset.customdata import CustomDataset
from transformers import BertTokenizer
import os
from torch.utils.data import DataLoader
import torch
from models.bert_classification import BERTClass

from torch import cuda
import numpy as np
from sklearn.metrics import accuracy_score

device = 'cuda' if cuda.is_available() else 'cpu'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
root_dir = "/home/charan/Documents/workspaces/python_workspaces/Data/ADL_Project"
classification = os.path.join(root_dir, "dept_classification.csv")
PATH = os.path.join(root_dir, "model_state_dict")

new_df = pd.read_csv(classification)
new_df.rename(columns={"DESCRIPTION": "desc", "DEPARTMENT_ID": "label"}, inplace=True)


def load_datasets(train_size=0.8):
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


training_loader, testing_loader = load_datasets()
model = BERTClass()
model.to(device)


def load_model(model_file):
    model.load_state_dict(torch.load(model_file))


model_file_path = os.path.join(root_dir, "model_state_dict2.pt")

load_model(model_file_path)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), PATH + str(epoch) + ".pt")


# for epoch in range(EPOCHS):
#     train(epoch)


def validation(epoch):
    model.eval()
    fin_targets = np.array([])
    fin_outputs = np.array([])
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets = np.append(fin_targets, np.argmax(targets.cpu().numpy(), axis=1))
            fin_outputs = np.append(fin_outputs, np.argmax(outputs.cpu().numpy(), axis=1))

    return fin_targets, fin_outputs


for epoch in range(EPOCHS):
    fin_targets, fin_outputs = validation(epoch)
    accuracy = accuracy_score(fin_outputs, fin_targets) * 100
    print('Epoch {} - accuracy {}'.format(epoch, accuracy))
