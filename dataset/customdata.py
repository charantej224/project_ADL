from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.features = dataframe.desc
        self.labels = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        desc_text = str(self.features[index])
        desc_text = " ".join(desc_text.split())

        inputs = self.tokenizer.encode_plus(
            desc_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        targets = torch.zeros(16)
        targets[self.labels[index]] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': targets
        }
