from include import *
from torch.autograd import Variable
import torch
import numpy as np


def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)

    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def metric(prob, label, thres = 0.5):
    shape = prob.shape
    prob_tmp = np.ones([shape[0], shape[1] + 1]) * thres
    prob_tmp[:, :shape[1]] = prob
    precision , top5 = top_n_np(prob_tmp, label)
    return  precision, top5


def top_n_np(preds, labels):
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)

    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i])/ (1.0*len(labels)))

    return re, top5

# def load_train_map(train_image_list_path = r'/data2/shentao/Projects/Kaggle_Whale/image_list/train_image_list.txt'):
#     f = open(train_image_list_path, 'r')
#     lines = f.readlines()
#     f.close()
#
#     label_dict = {}
#     for line in lines:
#         line = line.strip()
#         line = line.split(' ')
#         img_name = line[0]
#         index = int(line[1])
#         id = line[2]
#         label_dict[img_name] = [index,id]
#
#     return label_dict

# if __name__ == '__main__':
#     dict = load_train_map()
#     print(dict)