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

path = './kdd99/'
df = pd.read_csv(path + 'kddcup.data_10_percent_corrected')

df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'label'
]

# 数值列
number_col = df.select_dtypes(include=['number']).columns
# 分类变量
cat_col = df.columns.difference(number_col)
cat_col = cat_col.drop('label')
# 将分类变量筛选出来
df_cat = df[cat_col].copy()

# one-hot编码
one_hot_data = pd.get_dummies(df_cat, columns=cat_col)

# 将原数据的分类变量去掉
one_hot_df = pd.concat([df, one_hot_data],axis=1)
one_hot_df.drop(columns=cat_col, inplace=True)

normalized_df = utils.normalization(one_hot_df.copy(), number_col)

# 为不同的类别进行编码
labels = pd.DataFrame(df.label)
label_encoder = LabelEncoder()
enc_label = labels.apply(label_encoder.fit_transform)
normalized_df.label = enc_label
label_encoder.classes_
label_num = len(label_encoder.classes_)
#print(label_num)
#sys.exit()

data = normalized_df

X = data.drop(columns=['label'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

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