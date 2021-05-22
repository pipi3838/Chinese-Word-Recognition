import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json
import locale
# locale.setlocale(locale.LC_ALL, 'zh_TW.utf-8')

from models import Resnet, Net, Densenet
from ImgDataset import ImgDataset, CombineImgDataset
from tqdm import tqdm

seed = 5487
batch_size = 512
EPOCH = 100
device = 'cuda:0'
load_model_path = '/nfs/nas-5.1/wbcheng/CWR_models/Pretrain_Common/{6}_loss.pth'
save_model_path = '/nfs/nas-5.1/wbcheng/CWR_models/Desnet_All'

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

torch.manual_seed(seed)
np.random.seed(seed)

dir_path = './combine_dataset.txt'

with open('./word2label.json', 'r', encoding='utf-8') as f:
    word2label = json.load(f)

dataset = CombineImgDataset(dir_path, word2label)
# _, dataset = torch.utils.data.random_split(dataset, [len(dataset) - 100, 100])
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])
trainset_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valset_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

print('train ', len(train_set), 'eval ', len(valid_set), ' word label', len(word2label))

model = Densenet(4803)
model.load_state_dict(torch.load(load_model_path))
model.net.classifier = nn.Linear(model.net.classifier.in_features, len(word2label))
model = model.to(device)
# model = PretrainDensenet(len(word2label), load_model_path, 4803).to(device)
# model.load_state_dict(torch.load(load_model_path))

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def save_model(model, epoch, loss):
	filename = os.path.join(save_model_path,'{%d}_loss.pth'%(epoch))
	torch.save(model.state_dict(), filename)

min_loss = 1e8

for epoch in range(EPOCH):
    train_loss = 0
    total, match = 0.0, 0.0
    model.train()
    with tqdm(total=len(trainset_loader), desc='[Train]') as bar:
        for idx, batch in enumerate(trainset_loader): 
            imgs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            # torch.cuda.empty_cache()
            
            _, preds = torch.max(out, 1)
            match += sum((preds == labels)).item()
            total += labels.size(0)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()
            # raise Exception(img.shape, true_len.shape)
            bar.set_postfix({'loss' : '{0:1.5f}'.format(train_loss / (idx + 1)), "Acc" : '{0:.3f}'.format(match / total)})
            bar.update()
        # print(loss)
    train_loss /= len(trainset_loader)
    print ('[Train Epoch ' + str(epoch) + ']  Loss: ' + str(train_loss) + '   Acc: ' + str(match / total))

    val_loss = 0
    total, match = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(valset_loader), desc='[Val]') as bar:
            for idx, batch in enumerate(valset_loader): 
                imgs, labels = batch['image'].to(device), batch['label'].to(device)
                out = model(imgs)
                _, preds = torch.max(out, 1)
                match += sum((preds == labels)).item()
                total += labels.size(0)

                loss = criterion(out, labels)
                val_loss += loss.detach().cpu().item()

                bar.set_postfix({'loss' : '{0:1.5f}'.format(val_loss / (idx + 1)), "Acc" : '{0:.3f}'.format(match / total)})
                bar.update()
            # print(loss)
        val_loss /= len(valset_loader)
        print ('[Val Epoch ' + str(epoch) + ']  Loss: ' + str(val_loss) + '   Acc: ' + str(match / total))
        if val_loss < min_loss:
            print("Saved model with loss {:1.5f}".format(val_loss))
            min_loss = val_loss
            save_model(model, epoch, val_loss)
    print('\n')