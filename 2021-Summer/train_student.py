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
    def __init__(self, file_path, max_len):
        le = preprocessing.LabelEncoder()

        with open(file_path, 'rb') as f:
            self.X = torch.from_numpy(pickle.load(f)).float()
            self.Y = torch.from_numpy(le.fit_transform(pickle.load(f)))

        if max_len != -1:
            self.X = self.X[:max_len]
            self.Y = self.Y[:max_len]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Y': self.Y[idx]}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(valid_size, test_size, batch_size):

    network_dataset = Network_Dataset(file_path='dataset/preprocessed_dataset.pickle', max_len=args.max_len)
    
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


def run_epoch(epoch, teacher_model, student_model, optimizer, criterion, device, is_train, data_loader):

    total_loss = 0
    n_correct = 0
    n_total = 0
    
    if is_train: student_model.train()
    else: student_model.eval()

    for batch in data_loader:
        x, y = batch['X'].to(device), batch['Y'].to(device)
        batch_size = x.shape[0]

        teacher_pred = teacher_model(x).detach()
        student_pred = student_model(x)
        loss, metric = criterion(student_pred, y, teacher_pred)

        n_targets = len(y)
        n_total += n_targets 
        n_correct += (student_pred.argmax(-1) == y).long().sum().item()
        total_loss += loss.item() * n_targets

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    total_loss /= n_total
    print("Epoch {} / {} / Loss {:.3f} / Acc {:.1f}%".format(epoch, 'Train' if is_train else 'Valid', np.mean(total_loss), n_correct/n_total*100))

    return total_loss


def run_test(teacher_model, student_model, criterion, device, data_loader):

    total_loss = 0
    n_correct = 0
    n_total = 0
    
    student_model.eval()

    for batch in data_loader:
        x, y = batch['X'].to(device), batch['Y'].to(device)
        batch_size = x.shape[0]

        teacher_pred = teacher_model(x).detach()
        student_pred = student_model(x)
        loss, metric = criterion(student_pred, y, teacher_pred)

        n_targets = len(y)
        n_total += n_targets 
        n_correct += (student_pred.argmax(-1) == y).long().sum().item()
        total_loss += loss.item() * n_targets
            
    total_loss /= n_total
    print("Result / {} / Loss {:.3f} / Acc {:.1f}%".format('Test', np.mean(total_loss), n_correct/n_total*100))
    return total_loss


def distillation(y, labels, teacher_scores, T, alpha):
    criterion1 = nn.KLDivLoss()
    criterion2 = F.cross_entropy

    alpha1 = T*T * 2.0 + alpha
    alpha2 = 1.0 - alpha

    return alpha1 * criterion1(F.log_softmax(y/T), F.softmax(teacher_scores/T)) \
         + alpha2 * criterion2(y,labels)


def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
    

def distill_loss_batch(output, target, teacher_output):
    loss_b = distillation(output, target, teacher_output, T=20.0, alpha=0.7)
    metric_b = metric_batch(output, target)

    return loss_b, metric_b


def main(args):
    device = 'cuda' if torch.cuda.is_available and not args.use_cpu else 'cpu'
    print("using {}...".format(device))

    set_seed(0)
    
    train_dataloader, valid_dataloader, test_dataloader = load_dataset(args.valid_size, args.test_size, args.batch_size)
    print("loaded dataset..")

    teacher_model = MLP_Classifier(hidden_dim=args.teacher_hidden_dim).to(device)
    teacher_model.load_state_dict(torch.load("model/{}_best.pt".format(args.optimizer)))

    student_model = MLP_Classifier(hidden_dim=args.student_hidden_dim).to(device)

    mem_params = sum([param.nelement()*param.element_size() for param in teacher_model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in teacher_model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print("Teacher using", mem, "bytes..")
    
    mem_params = sum([param.nelement()*param.element_size() for param in student_model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in student_model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print("Student using", mem, "bytes..")

    criterion = distill_loss_batch
    optimizer = None

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr)
    elif args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'NAG':
        optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(student_model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSprop(student_model.parameters(), lr=args.lr, eps=1e-08)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, eps=1e-08)

    best_val_loss = np.inf
    t0 = time.time()

    for epoch in range(args.epochs):

        run_epoch(epoch, teacher_model, student_model, optimizer, criterion, device, is_train=True, data_loader=train_dataloader)

        with torch.no_grad():
            val_loss = run_epoch(epoch, teacher_model, student_model, None, criterion, device, is_train=False, data_loader=valid_dataloader)
            
        print("Epoch {} / {:.1f} seconds used.".format(epoch, time.time()-t0))
        t0 = time.time()
            
        if val_loss < best_val_loss:
            print("min_loss updated!")
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), "model/student_best.pt")


    student_model.load_state_dict(torch.load("model/student_best.pt"))
    run_test(teacher_model, student_model, criterion, device, data_loader=test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Samsung Research - Optimizer")

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--teacher_hidden_dim', type=int, default=256)
    parser.add_argument('--student_hidden_dim', type=int, default=128)

    parser.add_argument('--use_cpu', action="store_true")
    parser.add_argument('--max_len', type=int, default=10000)
    parser.add_argument('--optimizer', choices=['SGD', 'momentum', 'NAG', 'Adagrad', 'RMSProp', 'Adam'], default='Adagrad')

    args = parser.parse_args()
    
    main(args)