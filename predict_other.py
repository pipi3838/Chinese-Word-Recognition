# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json

from efficientnet_pytorch import EfficientNet

from ImgDataset import EffiDataset
from tqdm import tqdm

seed = 5487
batch_size = 128
EPOCH = 100
device = 'cuda:0'
load_model_path = '/nfs/nas-5.1/wbcheng/CWR_models/efficient/efficientnet_b3.pt'
torch.manual_seed(seed)
np.random.seed(seed)

model = EfficientNet.from_name('efficientnet-b3')
model._fc = nn.Linear(1536, 800)
# print(model._fc)
model.load_state_dict(torch.load(load_model_path))


dir_path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition/Origin'

# label2word = {val:key for key, val in word2label.items()}

word2label = dict()
with open('./idx2class.txt', 'rb') as f:
    lines = f.readlines()
    for idx, l in enumerate(lines):
        word2label[idx] = l.strip()
        # print(l.strip())
        # print(isinstance(l.strip(), str))
        # input()


dataset = EffiDataset(dir_path, word2label)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# _, dataset = torch.utils.data.random_split(dataset, [len(dataset) - 100, 100])
# train_set_size = int(len(dataset) * 0.8)
# valid_set_size = len(dataset) - train_set_size
# train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])
# trainset_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# valset_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

# print(len(train_set), len(valid_set))


total, match = 0.0, 0.0
store = defaultdict(dict)
with torch.no_grad():
    with tqdm(total=len(loader), desc='[Val]') as bar:
        for idx, batch in enumerate(loader): 
            imgs, labels = batch['image'].to(device), batch['label'].to(device)
            file_ids = batch['file_id']
            out = model(imgs)
            out = softmax(out)
            vals, preds = torch.max(out, 1)
            match += sum((preds == labels)).item()
            total += labels.size(0)

            bar.update()

print("Acc is {}".format(match / total))