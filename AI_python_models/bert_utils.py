import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

class Liar2Dataset(Dataset):
    def __init__(self, df_text: pd.DataFrame, df_meta: pd.DataFrame, tokenizer, max_length: int):
        self.df_text = df_text.reset_index(drop=True)
        self.df_meta = df_meta.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta_columns = list(df_meta.columns)

    def __len__(self):
        return len(self.df_text)

    def __getitem__(self, idx):
        text = self.df_text.loc[idx, 'text_for_bert']
        label = int(self.df_text.loc[idx, 'label'])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
  
        meta_tensor = torch.tensor(self.df_meta.loc[idx, self.meta_columns].values.astype('float32'), dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'metadata': meta_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_split_data(text_csv: str, meta_csv: str, test_size=0.2, val_size=0.5, random_state=200900):
    df_text = pd.read_csv(text_csv)
    df_meta = pd.read_csv(meta_csv)
    assert len(df_text) == len(df_meta)
    df_text_train, df_text_temp, df_meta_train, df_meta_temp = train_test_split(
        df_text, df_meta, test_size=test_size, stratify=df_text['label'], random_state=random_state
    )
    df_text_val, df_text_test, df_meta_val, df_meta_test = train_test_split(
        df_text_temp, df_meta_temp, test_size=val_size, stratify=df_text_temp['label'], random_state=random_state
    )
    return df_text_train, df_meta_train, df_text_val, df_meta_val, df_text_test, df_meta_test