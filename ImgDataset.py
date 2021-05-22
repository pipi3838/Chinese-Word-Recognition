# coding=utf-8
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from collections import defaultdict
import json
import numpy as np
from skimage.filters import threshold_otsu

class ImgDataset(Dataset):
    def __init__(self, dir_path, word2label=None, transform=None):
        super(ImgDataset, self).__init__()
        self.image_files = []
        self.labels = []
        self.words = []
        self.ids = []
        # Standard Preprocessing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((50, 50)),
                # transforms.Grayscale(),
                # lambda x: 1 - x,
                # transforms.Normalize(mean=[0.5], std=[0.5]),
                # transforms.RandomRotation([-20, 20]),
                # transforms.RandomResizedCrop((50, 50), scale=(0.7,1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else: self.transform = transform

        # word2label = defaultdict(lambda: len(word2label))
        for file in glob(os.path.join(dir_path, '*.jpg')):
            self.image_files.append(str(file))
            file_id = file.split('/')[-1]
            label = file_id[-5]
            self.labels.append(word2label[label])
            self.words.append(label) 
            self.ids.append(file_id)
        self.labels = torch.LongTensor(self.labels)
        # with open('./word2label.json', 'w', encoding='utf-8') as writer:
        #     writer.write(json.dumps(word2label, indent=4) + "\n")

    def __getitem__(self, idx):
        # img = Image.open(self.image_files[idx])
        img = Image.open(self.image_files[idx])
        img = self.transform(img)
        sample = {"image": img, "label": self.labels[idx], "word": self.words[idx], "file_id": self.ids[idx]}
        
        return sample

    def __len__(self):
        return len(self.labels)


class TransferImgDataset(Dataset):
    def __init__(self, dir_path, word2label=None):
        super(TransferImgDataset, self).__init__()
        self.image_files = []
        self.labels = []
        self.words = []
        self.ids = []
        # Standard Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            # transforms.Grayscale(),
            # lambda x: 1 - x,
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.RandomRotation([-10, 10]),
            # transforms.RandomResizedCrop((50, 50), scale=(0.9,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # word2label = defaultdict(lambda: len(word2label))
        for file in glob(os.path.join(dir_path, '*')):
            self.image_files.append(str(file))
            file_id = file.split('/')[-1]
            label = file_id.split('_')[0]
            self.labels.append(word2label[label])
            self.words.append(label) 
            self.ids.append(file_id)

        self.labels = torch.LongTensor(self.labels)
        # with open('./CommonWord2label.json', 'w', encoding='utf-8') as writer:
            # writer.write(json.dumps(word2label, indent=4) + "\n")

    def __getitem__(self, idx):
        # img = Image.open(self.image_files[idx])
        img = Image.open(self.image_files[idx]).convert("RGB")
        img = self.transform(img)
        sample = {"image": img, "label": self.labels[idx], "word": self.words[idx], "file_id": self.ids[idx]}
        
        return sample

    def __len__(self):
        return len(self.labels)

class CombineImgDataset(Dataset):
    def __init__(self, dir_path, word2label=None):
        super(CombineImgDataset, self).__init__()
        self.image_files = []
        self.labels = []
        self.words = []
        self.ids = []
        # Standard Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((50, 50)),
            # transforms.Grayscale(),
            # lambda x: 1 - x,
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.RandomResizedCrop((50, 50), scale=(0.9,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        with open(dir_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        
        for data in datas:
            img_path = data['img_path']
            label = data['label']
            self.image_files.append(img_path)
            self.ids.append(img_path.split('/')[-1])
            self.words.append(label)
            self.labels.append(word2label[label])

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, idx):
        # img = Image.open(self.image_files[idx])
        img = Image.open(self.image_files[idx].encode('utf-8')).convert("RGB")
        img = self.transform(img)
        sample = {"image": img, "label": self.labels[idx], "word": self.words[idx], "file_id": self.ids[idx]}
        
        return sample

    def __len__(self):
        return len(self.labels)

# dir_path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition'
# with open('./word2label.json', 'r', encoding='utf-8') as f:
#     word2label = json.load(f)

# trainset = ImgDataset(dir_path, word2label)
# loader = DataLoader(trainset, batch_size=4)

# batchs = next(iter(loader))
# img = batchs['image'][0]
# print(img)

