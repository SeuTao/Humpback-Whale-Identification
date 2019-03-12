from include import *
from torch.autograd import Variable
import torch
import numpy as np
from process.data_helper import *

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


def prob_to_csv_top5(prob, key_id, name):
    CLASS_NAME,_ = load_CLASS_NAME()

    prob = np.asarray(prob)
    print(prob.shape)

    top = np.argsort(-prob,1)[:,:5]
    word = []
    index = 0

    rs = []

    for (t0,t1,t2,t3,t4) in top:
        word.append(
            CLASS_NAME[t0] + ' ' + \
            CLASS_NAME[t1] + ' ' + \
            CLASS_NAME[t2])

        top_k_label_name = r''
        label = CLASS_NAME[t0]
        score = prob[index][t0]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t1]
        score = prob[index][t1]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t2]
        score = prob[index][t2]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t3]
        score = prob[index][t3]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t4]
        score = prob[index][t4]
        top_k_label_name += label + ' ' + str(score) + ' '

        rs.append(top_k_label_name)
        index += 1

    pd.DataFrame({'key_id':key_id, 'word':rs}).to_csv( '{}.csv'.format(name), index=None)
