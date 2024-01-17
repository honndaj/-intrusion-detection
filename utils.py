import os

import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index])
        y = torch.tensor(self.y.iloc[index])
        return X, y

def train(train_data, batch_size, device, model, optimizer, loss_fn, epochs):

    losses = []
    iter = 0
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    for epoch in range(epochs):
        print(f"epoch {epoch+1}\n-----------------")
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            X = X.reshape(X.shape[0], 1, X.shape[1])

            y_pred = model(X)
            loss = loss_fn(y_pred, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"loss: {loss.item()}\t[{(i+1)*len(X)}/{len(train_data)}]")

                iter += 1
                losses.append(loss.item())

    return losses, iter


def test(test_data, batch_size, device, model, loss_fn):
    """
    计算准确率和平均损失
    """
    positive = 0
    negative = 0
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    with torch.no_grad():
        iter = 0
        loss_sum = 0
        for X, y in test_dataloader:
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            y_pred = model(X)
            # print(f"y_pred: {y_pred.shape}, y: {y.shape}")
            loss = loss_fn(y_pred, y.long())
            loss_sum += loss.item()
            iter += 1
            for item in zip(y_pred, y):
                if torch.argmax(item[0]) == item[1]:
                    positive += 1
                else:
                    negative += 1
    acc = positive / (positive + negative)
    avg_loss = loss_sum / iter
    print("Accuracy:", acc)
    print("Average Loss:", avg_loss)

def test_per_class(test_data, batch_size, device, model, loss_fn):
    """
    ，以及每一个类别的准确率
    """
    positive = 0
    negative = 0
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    per_class_sum = [0 for i in range(10)]
    per_class_true = [0 for i in range(10)]
    with torch.no_grad():
        iter = 0
        loss_sum = 0
        for X, y in test_dataloader:
            X, y = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
            X = X.reshape(X.shape[0], 1, X.shape[1])
            y_pred = model(X)
            # print(f"y_pred: {y_pred.shape}, y: {y.shape}")
            loss = loss_fn(y_pred, y.long())
            loss_sum += loss.item()
            iter += 1
            for item in zip(y_pred, y):
                per_class_sum[int(item[1])] += 1
                if torch.argmax(item[0]) == item[1]:
                    positive += 1
                    per_class_true[int(item[1])] += 1
                else:
                    negative += 1
    acc = positive / (positive + negative)
    avg_loss = loss_sum / iter
    print("Accuracy:", acc)
    print("Average Loss:", avg_loss)
    for i in range(10) :
       print(f"class:{i}\t{per_class_true[i] / per_class_sum[i]}") 

def loss_value_plot(losses, iter):
    plt.figure()
    plt.plot([i for i in range(1, iter+1)], losses)
    plt.xlabel('Iterations (×100)')
    plt.ylabel('Loss Value')

def normalization(df,col):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
    return df
