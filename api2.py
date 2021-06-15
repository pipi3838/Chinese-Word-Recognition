from argparse import ArgumentParser
import base64
import datetime
import hashlib
import time
from collections import defaultdict
import json

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# from models import Densenet
from efficientnet_pytorch import EfficientNet


app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'mhliu@nlg.csie.ntu.edu.tw'          #
SALT = '5487878'                        #
#########################################

seed = 5487
device = 'cuda:0'
if not torch.cuda.is_available(): device='cpu'

# with open('./word2label.json', 'r', encoding='utf-8') as f:
#     word2label = json.load(f)
# label2word = {val:key for key, val in word2label.items()}

# label2word2 = dict()
# with open('./idx2class.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     for idx, l in enumerate(lines):
#         label2word2[idx] = l.strip()

label2word3 = dict()
with open('./idx2class_revised.dat', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, l in enumerate(lines):
        label2word3[idx] = l.strip()

# first_model = '{2}_loss.pth'
second_model = 'gridmask2.pt'
# third_model = 'efficientnet_b3.pt'
forth_model = 'affine.pt'

# model1 = Densenet(len(word2label)).to(device)
# model1.load_state_dict(torch.load(first_model, map_location=torch.device('cpu')))
# model1.eval()       

model2 = EfficientNet.from_name('efficientnet-b3')
model2._fc = nn.Linear(1536, 800)
model2.load_state_dict(torch.load(second_model, map_location=torch.device('cpu')))
model2.eval()

# model3 = EfficientNet.from_name('efficientnet-b3')
# model3._fc = nn.Linear(1536, 800)
# model3.load_state_dict(torch.load(third_model, map_location=torch.device('cpu')))
# model3.eval()

model4 = EfficientNet.from_name('efficientnet-b3')
model4._fc = nn.Linear(1536, 800)
model4.load_state_dict(torch.load(forth_model, map_location=torch.device('cpu')))
model4.eval()

softmax = nn.Softmax(dim=1)

# def preprocess_img(img):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     _, thre_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     thre_img = cv2.fastNlMeansDenoising(thre_img, h=13, searchWindowSize=7)
    
#     thre_img = cv2.fastNlMeansDenoising(thre_img, h=13, searchWindowSize=7)
#     kernel = np.ones((3,3), np.uint8)
#     thre_img = cv2.erode(thre_img, kernel, iterations = 1)

#     rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
#     dilate = cv2.dilate(thre_img, rect_kernel, iterations = 1)
    
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if len(contours) == 0: bound = cv2.resize(img, (50,50))
#     else:
#         max_area = -1
#         bx, by, bw, bh = None, None, None, None
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if w * h > max_area:
#                 bx, by, bw, bh = x, y, w, h
#                 max_area = w * h

#         bound = cv2.resize(img[by:by+bh, bx:bx+bw], (50, 50))
    
#     _, bound = cv2.threshold(bound, 0, 255, cv2.THRESH_OTSU)
#     bound = cv2.cvtColor(bound, cv2.COLOR_GRAY2BGR)

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     bound = transform(bound)
#     return bound

def preprocess3_img(img):
    img = img[:,:,::-1]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img

def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image, model, preprocess_func, mapping):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    # prediction = '陳'
    img = preprocess_func(image).to(device)
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        out = model(img)
        out = softmax(out)
        
        pred_matrix = defaultdict()
        for idx, val in enumerate(out[0]):
            pred_matrix[mapping[idx]] = val.item()

    pred_matrix = {word: pred_matrix[word] for word in sorted(pred_matrix.keys())}
    return pred_matrix
        # val, pred = torch.max(out, 1)
        
        # print(pred, val)
        # prediction = mapping[int(pred.item())]
    
    ####################################################
    # if val < 0.2: prediction = 'isnull'
    
    # if _check_datatype_to_string(prediction):
    #     return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
    cv2.imwrite('./image/{}_img.jpg'.format(ts), image)
    try:
        # answer1 = predict(image, model1, preprocess_img, label2word)
        answer2 = predict(image, model2, preprocess3_img, label2word3)
        # answer3 = predict(image, model3, preprocess3_img, label2word2)
        answer4 = predict(image, model4, preprocess3_img, label2word3)

        pred_ans, pred_val = None, -1
        answer_list = [answer2, answer4]
        # weight = [2, 1, 1, 0.6]
        weight = [1.2, 1]
        for word in answer2.keys():
            val = 0
            for a_id, ans in enumerate(answer_list):
                val += ans[word] * weight[a_id]
            if val > pred_val:
                pred_ans = word
                pred_val = val

        # for word, word2, val1, val2 in zip():
        #     if val1 + val2 > pred_val: 
        #         assert(word == word2)
        #         pred_ans = word
        #         pred_val = val1 + val2
        
        # if pred_val < 0.2 * sum(weight): pred_ans = 'isnull'
        if _check_datatype_to_string(pred_ans): answer = pred_ans

    except TypeError as type_error:
        # You can write some log...
        print('prediction type error')
        raise type_error
    except Exception as e:
        # You can write some log...
        print('can not predict image')
        raise e
    server_timestamp = time.time()
    with open('./result/{}_img.txt'.format(ts), 'w') as f:
        f.write(answer)
    print(answer)
    res =   jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})
    
    return res

if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()
    
    app.run(host='0.0.0.0', port=options.port, debug=options.debug)
