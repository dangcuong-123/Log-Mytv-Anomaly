from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import os
from datetime import datetime


class LogDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_encoder = LabelEncoder()
        self.dataframe['label'] = self.label_encoder.fit_transform(
            dataframe['label'])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['value']
        label = row['label']

        tokens = self.tokenizer(text)
        # Ensure no token exceeds vocab_size - 1
        tokens = [min(token, vocab_size - 1) for token in tokens]
        tokens = tokens[:self.max_seq_len]
        tokens += [0] * (self.max_seq_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def tokenizer(text):
    # Simple tokenization example
    return [ord(c) for c in text]


class SimplePositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super(SimplePositionalEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        positions = positions.long()
        pos_embeds = self.position_embedding(positions)
        return x + pos_embeds


class FeedforwardModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_len):
        super(FeedforwardModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = SimplePositionalEmbedding(
            vocab_size, embedding_dim, max_seq_len)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x = x.long()
        x = x.to(device)
        x = self.embedding(x)
        x = self.positional_embedding(x)

        x = x.mean(dim=1)
        x = self.flatten(x)
        return x


df_train = pd.read_csv(
    r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_api\train_data.csv", sep='|')
df_val = pd.read_csv(
    r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_api\test_data.csv", sep='|')
df_test = pd.read_csv(
    r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_api\test_data.csv", sep='|')

max_seq_len = 512
vocab_size = 512
embedding_dim = 128
hidden_dim = 128

dataset_train = LogDataset(df_train, tokenizer, max_seq_len)
dataset_val = LogDataset(df_val, tokenizer, max_seq_len)
dataset_test = LogDataset(df_test, tokenizer, max_seq_len)

train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)

embedding_model = FeedforwardModel(
    vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, max_seq_len=max_seq_len)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(embedding_model.parameters(), lr=1)


def train(epoch):
    embedding_model.train()
    train_loss = 0.
    train_recall = 0.
    train_f1 = 0.
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        for batch_idx, data in enumerate(train_bar, 0):
            # data = data.cuda()

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = embedding_model(inputs)

            loss = criterion(outputs, targets)

            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    train_loss /= len(train_loader)
    return train_loss


def val(epoch):
    embedding_model.eval()
    val_loss = 0.
    val_recall = 0.
    val_f1 = 0.
    for batch_idx, data in tqdm(enumerate(val_loader), desc="Validation"):
        with torch.no_grad():
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = embedding_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    return val_loss


def inference(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels


def evaluate(predictions, true_labels):
    # Compute accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy:.4f}')

    # Classification report (for more detailed metrics)
    report = classification_report(true_labels, predictions)
    print(f'Classification Report:\n{report}')


# Training loop
num_train_epochs = 250
# total number of training steps
t_total = len(train_loader) * num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=1000, num_training_steps=t_total)

best_loss = -1
model_name = 'random_test'

for epoch in range(1, num_train_epochs+1):
    print("Epoch:", epoch)
    train_loss = train(epoch)
    val_loss = val(epoch)

    print('Train Epoch: {} \tTrain Loss: {:.6f} \tValidation Loss: {:.6f} \tLearning rate: {}'.format(
        epoch, train_loss, val_loss, lr_scheduler.get_last_lr()))

    all_predictions, all_labels = inference(
        embedding_model, test_loader, device)
    evaluate(all_predictions, all_labels)

    if val_loss < best_loss or best_loss < 0:
        best_loss = val_loss
        print(f"Saving best model, loss: {best_loss}")
        if len(os.listdir('./weights')) > 0 and epoch > 1:
            os.system('rm {}'.format(
                f'./weights/{best_model_name}'))
        best_model_name = "{}_{}.pth".format(
            model_name, datetime.now().strftime("%Y-%m-%d"))
        torch.save(embedding_model, os.path.join(
            './weights', best_model_name))
