from dataset.customdata import CustomDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_datasets(classification_dataframe, train_size=0.8, number_of_classes=16):
    train_dataset = classification_dataframe.sample(frac=train_size, random_state=200)
    test_dataset = classification_dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(classification_dataframe.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, number_of_classes)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, number_of_classes)

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
