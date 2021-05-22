# coding=utf-8
import os
import numpy as np
import json
from collections import defaultdict
import csv
import numpy as np
import json
from glob import glob
import cv2

dir_path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition'

with open('./word2label.json', 'r', encoding='utf-8') as f:
    word2label = json.load(f)

keys = list(word2label.keys())
store = defaultdict(list)

inside, total = 0, 0

for img_path in glob(os.path.join(dir_path, 'Origin', '*')):
    img_name = img_path.split('/')[-1]
    label = img_name[-5]
    print(img_name, label)
    print(label)
    break
    input()

for img_path in glob(os.path.join(dir_path, 'Pretrain_Common_Modified', '*')):
    img_name = img_path.split('/')[-1]
    label = img_name.split('_')[0]
    print(label, type(label))
    if label in keys: inside += 1
    store[label].append(img_name)
    print(label)
    total += 1
    input()

# with open('./word2label.json', 'r', encoding='utf-8') as f:
#     word2label = json.load(f)

# label2word = {val:key for key, val in word2label.items()}


# path = '/nfs/nas-5.1/wbcheng/Chinese_Word_Recognition/TransferImg'
# for dir_path in glob(os.path.join(path, '*')):
#     for img in glob(os.path.join(dir_path, '*')):
#         label = img.split('/')[-1].split('_')[0]
#         label = label.encode('utf-8', 'ignore').decode('utf-8')
#         print(label, end=' ')
#         break
# with open('./true_cases', 'rb') as f:
#     data = np.load(f)

# print(len(label2word))

# thres = 0.5

# store = defaultdict(list)
# for img_name, val, pred in data:
#     if float(val) > thres: store[pred].append([img_name, val])

# print(len(store))


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

