# coding=utf-8
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import json
from collections import defaultdict
import numpy as np
import json
import cv2
from PIL import Image
from glob import glob

from models import Densenet
from ImgDataset import ImgDataset, CombineImgDataset
from tqdm import tqdm

seed = 5487
batch_size = 128
device = 'cuda:0'
if not torch.cuda.is_available(): device='cpu'

load_model_path = '/nfs/nas-5.1/wbcheng/CWR_models/Desnet_All/{2}_loss.pth'
torch.manual_seed(seed)
np.random.seed(seed)
data_path = './combine_dataset.txt'

with open('./word2label.json', 'r', encoding='utf-8') as f:
    word2label = json.load(f)
label2word = {val:key for key, val in word2label.items()}


# dataset = CombineImgDataset(data_path, word2label)
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Densenet(len(word2label))
model.load_state_dict(torch.load(load_model_path))
model = model.to(device)

model.eval()
softmax = nn.Softmax(dim=1)

def preprocess_img(img):
    _, thre_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    thre_img = cv2.fastNlMeansDenoising(thre_img, h=13, searchWindowSize=7)
    
    thre_img = cv2.fastNlMeansDenoising(thre_img, h=13, searchWindowSize=7)
    kernel = np.ones((3,3), np.uint8)
    thre_img = cv2.erode(thre_img, kernel, iterations = 1)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilate = cv2.dilate(thre_img, rect_kernel, iterations = 1)
    
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0: bound = cv2.resize(img, (50,50))
    else:
        max_area = -1
        bx, by, bw, bh = None, None, None, None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_area:
                bx, by, bw, bh = x, y, w, h
                max_area = w * h

        bound = cv2.resize(img[by:by+bh, bx:bx+bw], (50, 50))
    
    _, bound = cv2.threshold(bound, 0, 255, cv2.THRESH_OTSU)
    bound = cv2.cvtColor(bound, cv2.COLOR_GRAY2BGR)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    bound = transform(bound)
    return bound

def predict_img(img):
    img = preprocess_img(img).to(device)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        out = model(img)
        out = softmax(out)
        val, pred = torch.max(out, 1)
        print(pred, val)
        pred = label2word[int(pred.item())]
    print('Finish prediction')
    return pred

path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition/'
for img_path in glob(os.path.join(path, 'Origin', '*.jpg')):
    img = cv2.imread(img_path, 0)
    assert(img is not None)
    print('finish loading image')
    pred = predict_img(img)
    print('pred: ' , pred, 'ans: ', img_path.split('/')[-1][-5])

    input()

# Predict with dataloader (No need to called preprocess_img)
# total, match = 0.0, 0.0
# store = defaultdict(dict)
# with torch.no_grad():
#     with tqdm(total=len(loader), desc='[Val]') as bar:
#         for idx, batch in enumerate(loader): 
#             imgs, labels = batch['image'].to(device), batch['label'].to(device)
#             file_ids = batch['file_id']
#             out = model(imgs)
#             out = softmax(out)
#             vals, preds = torch.max(out, 1)
#             match += sum((preds == labels)).item()
#             total += labels.size(0)

#             for val, pred, label, file_id in zip(vals, preds, labels, file_ids):
#                 correct = pred == label
#                 store[file_id] = {
#                     'correct': correct.item(),
#                     'val': float(val.item()),
#                     'pred_label': label2word[int(pred.item())],
#                 }

#             bar.update()

# print("Acc is {}".format(match / total))
# with open('./combine_predict.json', 'w') as f:
#     json.dump(store, f)

