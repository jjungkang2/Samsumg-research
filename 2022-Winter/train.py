import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import random
import pickle
from sklearn import preprocessing

from model import MLP_Classifier

class Network_Dataset(Dataset):
    def __init__(self, file_path, max_len, augmentation='None', new_data_len=0):
        le = preprocessing.LabelEncoder()

        with open(file_path, 'rb') as f:
            self.X = torch.from_numpy(pickle.load(f)).float()
            self.Y = torch.from_numpy(le.fit_transform(pickle.load(f)))
            self.Y = self.Y.to(torch.int64)

        if max_len != -1:
            indices = torch.randperm(self.Y.size(0))

            self.X = torch.index_select(self.X, dim=0, index=indices)
            self.Y = torch.index_select(self.Y, dim=0, index=indices)

            self.X = self.X[:max_len]
            self.Y = self.Y[:max_len]

        if augmentation == 'Mixup':
            X_new = []
            Y_new = []
            
            data_len = self.Y.size(0)

            for _ in range(new_data_len):
                data1_idx = random.randint(0, data_len-1)
                same_classes = (self.Y == self.Y[data1_idx]).nonzero(as_tuple=True)[0]
                data2_idx = same_classes[random.randint(0, same_classes.size(0)-1)]

                X_new.append((self.X[data1_idx] + self.X[data2_idx]) / 2)
                Y_new.append(self.Y[data1_idx])

            self.X = torch.cat([self.X, torch.stack(X_new)], dim=0)
            self.Y = torch.cat([self.Y, torch.stack(Y_new)], dim=0)
        
        elif augmentation == 'Delete':
            X_new = []
            Y_new = []
            
            data_len = self.Y.size(0)
            feature_len = self.X.size(1)

            for _ in range(new_data_len):
                data_idx = random.randint(0, data_len-1)

                delete_indices = torch.bernoulli(torch.ones(feature_len) * args.probability)
                data = torch.where(delete_indices==0, self.X[data_idx], torch.zeros(feature_len))

                X_new.append(data)
                Y_new.append(self.Y[data_idx])

            self.X = torch.cat([self.X, torch.stack(X_new)], dim=0)
            self.Y = torch.cat([self.Y, torch.stack(Y_new)], dim=0)

        elif augmentation == 'Modify':
            X_new = []
            Y_new = []
            
            data_len = self.Y.size(0)
            feature_len = self.X.size(1)

            for _ in range(new_data_len):
                data_idx = random.randint(0, data_len-1)

                delete_indices = torch.bernoulli(torch.ones(feature_len) * args.probability)
                data = torch.where(delete_indices==0, self.X[data_idx], self.X[data_idx] + torch.randn(feature_len)/2)

                X_new.append(data)
                Y_new.append(self.Y[data_idx])

            self.X = torch.cat([self.X, torch.stack(X_new)], dim=0)
            self.Y = torch.cat([self.Y, torch.stack(Y_new)], dim=0)

        elif augmentation == 'None':
            pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Y': self.Y[idx]}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(valid_size, test_size, batch_size):

    network_dataset = Network_Dataset(file_path='dataset/preprocessed_dataset.pickle', 
                                      max_len=args.max_len,
                                      augmentation=args.augmentation,
                                      new_data_len=args.new_data_len)
    
    total_length = len(network_dataset)

    valid_len = round(total_length*valid_size)
    test_len = round(total_length*test_size)
    train_len = total_length - valid_len - test_len

    print("train {} / valid {} / test {}...".format(train_len, valid_len, test_len))

    train_dataset, valid_dataset, test_dataset = random_split(network_dataset, [train_len, valid_len, test_len])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, drop_last = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

    return train_dataloader, valid_dataloader, test_dataloader


def run_epoch(epoch, model, optimizer, criterion, device, is_train, data_loader):

    total_loss = 0
    n_correct = 0
    n_total = 0
    
    if is_train: model.train()
    else: model.eval()

    for batch in data_loader:
        x, y = batch['X'].to(device), batch['Y'].to(device)
        batch_size = x.shape[0]

        pred = model(x)
        loss = criterion(pred.reshape(batch_size,-1), y.reshape(batch_size))

        n_targets = len(y)
        n_total += n_targets 
        n_correct += (pred.argmax(-1) == y).long().sum().item()
        total_loss += loss.item() * n_targets

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    total_loss /= n_total
    print("Epoch {} / {} / Loss {:.3f} / Acc {:.1f}%".format(epoch, 'Train' if is_train else 'Valid', np.mean(total_loss), n_correct/n_total*100))

    return total_loss


def run_test(model, criterion, device, data_loader):

    total_loss = 0
    n_correct = 0
    n_total = 0
    
    model.eval()

    for batch in data_loader:
        x, y = batch['X'].to(device), batch['Y'].to(device)
        batch_size = x.shape[0]

        pred = model(x)
        loss = criterion(pred.reshape(batch_size,-1), y.reshape(batch_size))

        n_targets = len(y)
        n_total += n_targets 
        n_correct += (pred.argmax(-1) == y).long().sum().item()
        total_loss += loss.item() * n_targets
            
    total_loss /= n_total
    print("Result / {} / Loss {:.3f} / Acc {:.1f}%".format('Test', np.mean(total_loss), n_correct/n_total*100))
    return total_loss


def main(args):
    device = 'cuda' if torch.cuda.is_available and not args.use_cpu else 'cpu'
    print("using {}...".format(device))

    set_seed(0)
    
    train_dataloader, valid_dataloader, test_dataloader = load_dataset(args.valid_size, args.test_size, args.batch_size)
    print("loaded dataset..")

    model = MLP_Classifier(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

    best_val_loss = np.inf

    for epoch in range(args.epochs):

        run_epoch(epoch, model, optimizer, criterion, device, is_train=True, data_loader=train_dataloader)

        with torch.no_grad():
            val_loss = run_epoch(epoch, model, None, criterion, device, is_train=False, data_loader=valid_dataloader)
            
        if val_loss < best_val_loss:
            print("min_loss updated!")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model/{}_best.pt".format(args.augmentation))

    model.load_state_dict(torch.load("model/{}_best.pt".format(args.augmentation)))
    run_test(model, criterion, device, data_loader=test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Samsung Research - Data Augmentation")

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--use_cpu', action="store_true")
    parser.add_argument('--max_len', type=int, default=1000)

    parser.add_argument('--augmentation', choices=['None', 'Mixup', 'Delete', 'Modify'], default='None')
    parser.add_argument('--new_data_len', type=int, default=0)
    parser.add_argument('--probability', type=float, default=0.1)

    args = parser.parse_args()
    
    main(args)