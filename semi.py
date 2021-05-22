import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict
import csv
import numpy as np
import json

from models import Resnet, Net, Densenet
from ImgDataset import ImgDataset
from tqdm import tqdm

seed = 5487
batch_size = 128
EPOCH = 100
device = 'cuda:0'
save_model_path = '/nfs/nas-5.1/wbcheng/CWR_models/DenseNet121'
torch.manual_seed(seed)
np.random.seed(seed)

dir_path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition'
with open('./word2label.json', 'r', encoding='utf-8') as f:
    word2label = json.load(f)

label2word = {val:key for key, val in word2label.items()}

dataset = ImgDataset(dir_path, word2label)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Densenet(len(word2label)).to(device)
model.load_state_dict(torch.load('/nfs/nas-5.1/wbcheng/CWR_models/DenseNet121/{17}_loss{1.6857}.pth')['net'])
store = defaultdict(dict)

total, match = 0.0, 0.0
model.eval()
softmax = nn.Softmax(dim=1)

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

            for val, pred, label, file_id in zip(vals, preds, labels, file_ids):
                correct = pred == label
                store[file_id] = {
                    'correct': correct.item(),
                    'val': float(val.item()),
                    'pred_label': label2word[int(pred.item())],
                }

                # print(type(float(val.item())))
            bar.update()

# with open('./all_cases', 'w') as f:
    # json.dump(store, f)

# false_case = [[key, val['val'], val['pred_label']] for key, val in store.items() if not val['correct']]
# false_case = sorted(false_case, key=lambda c: -c[1])

# with open('./false_cases', 'wb') as f:
#     np.save(f, false_case)

# true_case = [[key, val['val'], val['pred_label']] for key, val in store.items() if val['correct']]
# true_case = sorted(true_case, key=lambda c: c[1])

# f = open('./true_cases', 'wb')
# np.save(f, true_case)
# f.close()


# with open('./pred_list.csv', 'w') as f:
#     writer = csv.writer(f)
#     for idx, (key, val) in enumerate(store.items()):
#         writer.writerow([idx, val['correct'], val['val']])

