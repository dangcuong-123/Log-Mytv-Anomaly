import torch
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from model.model_roberta import RobertaClass
from data.log_cdn import LogCdn, load_data
from sklearn.metrics import classification_report
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def calcuate_accuracy(preds, targets):
    n_correct = (preds == targets).sum().item()
    return n_correct


def calculate_f1_recall(preds, targets):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average='macro')
    return recall, f1


def train(epoch):
    model.train()
    train_loss = 0.
    train_recall = 0.
    train_f1 = 0.
    with tqdm(training_loader, desc=f"Train Epoch {epoch}") as train_bar:
        for batch_idx, data in enumerate(train_bar, 0):
            # data = data.cuda()
            optimizer.zero_grad()
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_function(outputs, targets)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            recall, f1 = calculate_f1_recall(big_idx, targets)
            train_recall += recall
            train_f1 += f1

            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    train_loss /= len(training_loader)
    train_recall /= len(training_loader)
    train_f1 /= len(training_loader)
    return train_loss, train_recall, train_f1


def val(epoch):
    #    metric = load_metric("seqeval")
    model.eval()
    val_loss = 0.
    val_recall = 0.
    val_f1 = 0.
    val_list_predict = []
    val_list_target = []

    for batch_idx, data in tqdm(enumerate(valid_loader), desc="Validation"):
        with torch.no_grad():
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.long)
            # Flatten the targets tensor to match the input tensor size
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            val_loss += loss.item()

            big_val, big_idx = torch.max(outputs.data, dim=1)
            recall, f1 = calculate_f1_recall(big_idx, targets)
            val_recall += recall
            val_f1 += f1
            val_list_predict.extend(big_idx.tolist())
            val_list_target.extend(targets.tolist())

    val_loss /= len(valid_loader)
    val_recall /= len(valid_loader)
    val_f1 /= len(valid_loader)
    return val_loss, val_recall, val_f1, val_list_predict, val_list_target


if __name__ == '__main__':
    LEARNING_RATE = 1e-05
    MAX_LEN = 256
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_data = load_data(
        r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_cdn\training.csv", is_train=True)
    test_data = load_data(
        r"C:\Users\MSI\Desktop\RA\MyTvAnamalyDetection\log_mytv_anomaly\data_training\log_cdn\test.csv", is_train=False)
    training_set = LogCdn(train_data, tokenizer, MAX_LEN)
    valid_set = LogCdn(test_data, tokenizer, MAX_LEN)

    model = RobertaClass()
    model.to(DEVICE)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=5e-4)

    # Set num epochs
    num_train_epochs = 4
    # total number of training steps
    t_total = len(training_loader) * num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=1000, num_training_steps=t_total)

    best_loss = -1
    model_name = 'roberta_test'

    for epoch in range(1, num_train_epochs+1):
        print("Epoch:", epoch)
        train_loss, train_recall, train_f1 = train(epoch)
        val_loss, val_recall, val_f1, val_list_predict, val_list_target = val(
            epoch)

        print('Train Epoch: {} \tTrain Loss: {:.6f} \tValidation Loss: {:.6f} \tLearning rate: {}'.format(
            epoch, train_loss, val_loss, lr_scheduler.get_last_lr()))
        print(f"Train Recall: {train_recall}")
        print(f"Train F1: {train_f1}")
        print(f"Validation Recall: {val_recall}")
        print(f"Validation F1: {val_f1}")
        print("classification_report\n", classification_report(
            val_list_target, val_list_predict))

        if val_loss < best_loss or best_loss < 0:
            best_loss = val_loss
            print(f"Saving best model, loss: {best_loss}")
            if len(os.listdir('./weight')) > 0 and epoch > 1:
                os.system('rm {}'.format(f'./weight/{best_model_name}'))
            best_model_name = "{}_{}.pth".format(
                model_name, "_log_api_", datetime.now().strftime("%Y-%m-%d"))
            torch.save(model, os.path.join('./weight', best_model_name))
