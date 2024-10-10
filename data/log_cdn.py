# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder  # Thêm import
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import warnings
warnings.filterwarnings("ignore")


def load_data(file_path, is_train=True):
    if (is_train):
        data = pd.read_csv(file_path)
        # Tạo dict chuyển đổi
        conversion_dict = {"Bình thường": 0,
                           "Bất thường trong log": 1,
                           "Log sai định dạng": 2,
                           "hacker tấn công": 3}

        # Chuyển đổi giá trị của cột label_name theo dict và giữ nguyên các giá trị không có trong dict
        data['label'] = data['label'].map(
            conversion_dict).fillna(data['label'])
        return data
    else:
        data = pd.read_csv(file_path, header=None, names=['value'], sep='|')
        return data


class LogCdn(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.value
        if 'label' in dataframe.columns:  # Kiểm tra nếu có cột label
            self.targets = self.data.label
        else:  # Nếu không có cột label, gán targets là mảng gồm các phần tử 0
            self.targets = [0] * len(dataframe)
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


if __name__ == "__main__":
    MAX_LEN = 256

    train_data = load_data(
        r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_cdn\training.csv", is_train=True)
    test_data = load_data(
        r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_cdn\test.csv", is_train=False)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    training_set = LogCdn(train_data, tokenizer, MAX_LEN)
    valid_set = LogCdn(test_data, tokenizer, MAX_LEN)
