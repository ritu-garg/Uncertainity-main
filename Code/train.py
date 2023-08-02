from torch import nn
import torchvision.transforms as transforms
import numpy as np
import copy
from datasets import load_dataset, Image
import torch
import tensorflow_datasets as tfds 
import tensorflow as tf
import argparse
from torch.optim import Adam
import os

best_acc = 0

from utils import *
from model import *

def train(dataloader):
    train_loss = 0
    total = 0
    correct = 0
    model.train()
    for batch_idx, data in enumerate(dataloader):
        inputs = data["image"]
        # batch["image"] = np.transpose(batch["image"],(0,3,1,2))
        # batch["image"] = torch.from_numpy(batch["image"])
        inputs = inputs.numpy()
        inputs = np.transpose(inputs,(0,3,1,2))
        inputs = torch.from_numpy(inputs)
        
        # print(inputs.size())
        # inputs = np.transpose(inputs,(0,3,1,2))
        targets = data["label"] if args.mode == "fine" else data["coarse_label"]
        targets = torch.from_numpy(targets.numpy())
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs,mode="proj")
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(batch_idx+1),correct/total

def test(dataloader,mode="test"):
    global best_acc
    if mode == "test":
        model.eval()
    else:
        model.train()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            inputs = data["image"]
            # batch["image"] = np.transpose(batch["image"],(0,3,1,2))
            # batch["image"] = torch.from_numpy(batch["image"])
            inputs = inputs.numpy()
            inputs = np.transpose(inputs,(0,3,1,2))
            inputs = torch.from_numpy(inputs)
            
            # print(inputs.size())
            # inputs = np.transpose(inputs,(0,3,1,2))
            targets = data["label"] if args.mode == "fine" else data["coarse_label"]
            targets = torch.from_numpy(targets.numpy())


            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs,mode="proj")
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.isdir('../Models'):
            os.mkdir('../Models')
        torch.save(state, f'../Models/vit_cifar100_{args.mode}.t7')
        best_acc = acc

    return test_loss/(batch_idx+1),acc

def GetArgs():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--bs', default='512')
    parser.add_argument('--n_epochs', type=int, default='50')
    parser.add_argument('--prob',default=0)
    parser.add_argument('--dropout',default=0)
    parser.add_argument('--size',default=224)
    parser.add_argument("--mode",default="fine")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":

    args = GetArgs()

    batch_size,lr,EPOCHS,prob,dropout,size = int(args.bs),float(args.lr),int(args.n_epochs),float(args.prob),float(args.dropout),int(args.size)

    DATA_DIR = "../Data/"
    cifar10_ds_train = prepare_pure_dataset(tfds.load('cifar10', split='train', shuffle_files=False,data_dir = DATA_DIR), 10, shuffle=False, batch_size=batch_size)
    cifar10_ds_test = prepare_pure_dataset(tfds.load('cifar10', split='test', shuffle_files=False,data_dir = DATA_DIR), 10, shuffle=False, batch_size=batch_size)
    cifar100_ds_train = prepare_pure_dataset(tfds.load('cifar100', split='train', shuffle_files=False,data_dir = DATA_DIR), 100, shuffle=False, batch_size=batch_size)
    cifar100_ds_test = prepare_pure_dataset(tfds.load('cifar100', split='test', shuffle_files=False,data_dir = DATA_DIR), 100, shuffle=False, batch_size=batch_size)
    

    trainloader = cifar100_ds_train
    testloader = cifar100_ds_test

    early_stopper = EarlyStopper(patience=3, min_delta=0)
    num_classes = 100 if args.mode == "fine" else 20
    model = Neural_Img_Clf_model(dropout=dropout,no_classes=num_classes)
    model = torch.nn.DataParallel(model)
    model.cuda()
    optimizer = Adam(model.parameters(),lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    print("Training Started")
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss,train_acc = train(trainloader)
        test_loss,test_acc = test(testloader)
        if early_stopper.early_stop(test_acc):
            print("Early stopping")
            break
        print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f},\
            test loss: {test_loss:.4f}, test acc: {test_acc:.4f}, time: {time.time()-start:.2f}")
