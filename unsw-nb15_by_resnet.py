import utils
from net.resnet import *
import os
import time
import sys
from datetime import datetime

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

path = './UNSW-NB15/'
df_train = pd.read_csv(path + 'UNSW_NB15_training-set.csv')
df_test = pd.read_csv(path + 'UNSW_NB15_testing-set.csv')
df = pd.concat([df_train, df_test])
df = df.drop(['id', 'label'], axis=1)

#区分数值列和非数值列
number_col = df.select_dtypes(include=['number']).columns
cat_col = df.columns.difference(number_col)
cat_col = cat_col.drop('attack_cat')
df_cat = df[cat_col].copy()

# one-hot编码
one_hot_data = pd.get_dummies(df_cat, columns=cat_col)

# 将原数据的分类变量去掉
one_hot_df = pd.concat([df, one_hot_data],axis=1)
one_hot_df.drop(columns=cat_col, inplace=True)


normalized_df = utils.normalization(one_hot_df.copy(), number_col)

# 为不同的类别进行编码
labels = pd.DataFrame(df.attack_cat)
label_encoder = LabelEncoder()
enc_label = labels.apply(label_encoder.fit_transform)
normalized_df.attack_cat = enc_label
label_encoder.classes_
label_num = len(label_encoder.classes_)

data = normalized_df

X = data.drop(columns=['attack_cat'])
y = data['attack_cat']
X_train = X[0:df_train.shape[0]]
y_train = y[0:df_train.shape[0]]
X_test = X[df_train.shape[0]:]
y_test = y[df_train.shape[0]:]

    
train_data = utils.LoadData(X_train, y_train)
test_data = utils.LoadData(X_test, y_test)

batch_size = 256

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

resnet_model = ResNet(label_num)

epochs = 40
lr = 1e-4 
momentum = 0.9
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


resnet_model.to(device=device)

#X = torch.rand(size=(1, 1, 224), device = device)
#for layer in resnet_model:
    #X = layer(X)
    #print(layer.__class__.__name__,'output shape:\t', X.shape)


timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S_')
old_model_name = ''
new_model_name = timestamp + 'resnet_model.pth'

if old_model_name != '':
    resnet_model.load_state_dict(torch.load(path + old_model_name))
else:
    losses, iter = utils.train(train_data, batch_size, device, resnet_model, optimizer, loss_fn, epochs)
    torch.save(resnet_model.state_dict(), path + new_model_name)

    utils.loss_value_plot(losses, iter)
    plt.savefig(path + timestamp + 'resnet_loss.png')

utils.test(test_data, batch_size, device, resnet_model, loss_fn)