from torch import cuda
import torch
from models.bert_classification import BERTClass
import numpy as np
from sklearn.metrics import accuracy_score
import json
import time

device = 'cuda' if cuda.is_available() else 'cpu'


def setup_model(number_of_classes):
    bert_model = BERTClass(number_of_classes=number_of_classes)
    bert_model.to(device)
    return bert_model


def get_optimizer(bert_model):
    learning_rate = 1e-05
    adam_optimizer = torch.optim.Adam(params=bert_model.parameters(), lr=learning_rate)
    return adam_optimizer


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, training_loader, model, optimizer, model_directory):
    start = time.time()
    model.train()
    unique_ids = np.array([])
    train_targets = np.array([])
    train_outputs = np.array([])
    counter = 0
    total = len(training_loader)
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        train_targets = np.append(train_targets, np.argmax(targets.cpu().detach().numpy(), axis=1))
        train_outputs = np.append(train_outputs, np.argmax(outputs.cpu().detach().numpy(), axis=1))
        unique_ids = np.append(unique_ids, data['u_id'])
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter = counter + len(data)
        if counter % 100 == 0:
            print(f" Epoch - {epoch} - current training {counter} / {total}")

    torch.save(model.state_dict(), model_directory + '_' + str(epoch) + ".pt")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, train_targets, train_outputs, elapsed


def validation(epoch, testing_loader, model):
    start = time.time()
    model.eval()
    validation_targets = np.array([])
    validation_outputs = np.array([])
    unique_ids = np.array([])
    print(f'Epoch - Inference : {epoch}')
    counter = 0
    total = len(testing_loader)
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            validation_targets = np.append(validation_targets, np.argmax(targets.cpu().numpy(), axis=1))
            validation_outputs = np.append(validation_outputs, np.argmax(outputs.cpu().numpy(), axis=1))
            unique_ids = np.append(unique_ids, data['u_id'])
            counter = counter + len(data)
            if counter % 100 == 0:
                print(f" Epoch - {epoch} - current Inference {counter} / {total}")
    done = time.time()
    elapsed = (done - start) / 60
    return unique_ids, validation_targets, validation_outputs, elapsed


def start_epochs(training_loader, testing_loader, metrics_json, model_directory, epochs=3, number_of_classes=16):
    model = setup_model(number_of_classes)
    optimizer = get_optimizer(model)
    accuracy_map = {}
    for epoch in range(epochs):
        unique_ids, train_targets, train_outputs, train_time = train(epoch, training_loader, model, optimizer,
                                                                     model_directory)
        train_accuracy = accuracy_score(train_targets, train_outputs) * 100
        print('Epoch {} - accuracy {}'.format(epoch, train_accuracy))
        unique_ids, val_targets, val_outputs, inference_time = validation(epoch, testing_loader, model)
        validation_accuracy = accuracy_score(val_targets, val_outputs) * 100
        print('Epoch {} - accuracy {}'.format(epoch, validation_accuracy))
        accuracy_map["train_accuracy_" + str(epoch)] = train_accuracy
        accuracy_map["train_time_" + str(epoch)] = train_time
        accuracy_map["val_accuracy_" + str(epoch)] = validation_accuracy
        accuracy_map["val_time_" + str(epoch)] = inference_time

    with open(metrics_json, 'w') as f:
        json.dump(accuracy_map, f, indent=2)
        f.close()


def load_model(model_file, testing_loader, number_of_classes):
    model = setup_model(number_of_classes)
    model.load_state_dict(torch.load(model_file))
    unique_ids, val_targets, val_outputs, inference_time = validation(1, testing_loader, model)
    validation_accuracy = accuracy_score(val_targets, val_outputs) * 100
    print('Epoch {} - accuracy {} - Inference time {} '.format(1, validation_accuracy, inference_time))
    return unique_ids, val_outputs
