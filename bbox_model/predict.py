import numpy as np
import torch
from torch.nn import DataParallel
import cv2
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import sys

from config import Config
from kpda_parser import *
from cascade_pyramid_network import CascadePyramidNet
from utils import draw_keypoints
from bbox_model.helper.keypoint_encoder import KeypointEncoder
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def compute_keypoints(config, img0, net, encoder, doflip=False):
    img_h, img_w, _ = img0.shape
    # min size resizing
    scale = config.img_max_size / (max(img_w, img_h)*1.0)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)
    img = cv2.resize(img0, (img_w2, img_h2), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # channel, height, width
    img[[0, 2]] = img[[2, 0]]
    img = img / 255.0
    img = (img - config.mu) / (config.sigma*1.0)
    pad_imgs = np.zeros([1, 3, config.img_max_size, config.img_max_size], dtype=np.float32)
    pad_imgs[0, :, :img_h2, :img_w2] = img
    data = torch.from_numpy(pad_imgs)
    data = data.cuda(async=True)
    _, hm_pred = net(data)
    hm_pred = F.relu(hm_pred, False)
    hm_pred = hm_pred[0].data.cpu().numpy()

    if doflip:
        a = np.zeros_like(hm_pred)
        a[:, :, :img_w2 // config.hm_stride] = np.flip(hm_pred[:, :, :img_w2 // config.hm_stride], 2)
        for conj in config.conjug:
            a[conj] = a[conj[::-1]]
        hm_pred = a

    return hm_pred

def pts2bbox(pts):
    locs = np.array(pts)
    minimums = np.amin(locs, axis=0)
    maximums = np.amax(locs, axis=0)
    bbox = np.array([[max(minimums[0], 0), max(minimums[1], 0),
                      maximums[0], maximums[1]]], dtype=np.int64)
    return bbox[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--type', help='specify the clothing type', default='whale')
    parser.add_argument('-m', '--model', help='specify the model', default=None)
    parser.add_argument('-v', '--vis', help='whether visualize result', default=False)

    args = parser.parse_args(sys.argv[1:])
    config = Config(args.type)

    val_kpda = KPDA(config, 'infer')

    args.model = r'/data1/shentao/Projects/Kaggle_Whale2019_2nd_place_solution/bbox_model/checkpoints_se50/kpt_whale_best_val.ckpt'
    name = r'/data1/shentao/Projects/Kaggle_Whale2019_2nd_place_solution/bbox_model/checkpoints_se50/se50_flip_test'

    print('Testing: ' + config.type)
    print('Validation sample number: %d' % val_kpda.size())
    net = CascadePyramidNet(config)
    checkpoint = torch.load(args.model)  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()

    net = DataParallel(net)
    net.eval()
    encoder = KeypointEncoder()
    nes = []
    img_id = []
    keypoint_word = []

    Image, x0, y0, x1, y1 = [],[],[],[],[]

    for idx in tqdm(range(val_kpda.size())):
        img_path_ = val_kpda.get_image_path(idx)

        img_path = os.path.join(TST_IMGS_DIR, img_path_)
        if not os.path.exists(img_path):
            img_path = os.path.join(TRN_IMGS_DIR, img_path_)

        img0 = cv2.imread(img_path)  #X
        img0_flip = cv2.flip(img0, 1)
        img_h, img_w, _ = img0.shape

# ----------------------------------------------------------------------------------------------------------------------
        scale = config.img_max_size / (max(img_w, img_h)*1.0)
        with torch.no_grad():
            hm_pred = compute_keypoints(config, img0, net, encoder) #* pad_mask
            hm_pred2 = compute_keypoints(config, img0_flip, net, encoder, doflip=True) #* pad_mask

        x, y = encoder.decode_np(hm_pred + hm_pred2, scale, config.hm_stride, (img_w/2, img_h/2), method='maxoffset')
        keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
        bbox = pts2bbox(keypoints)

# ----------------------------------------------------------------------------------------------------------------------
        img_id.append(os.path.split(img_path)[1])
        pt_str = r''
        for i in range(4):
            x,y,v = keypoints[i]
            pt = str(x)+'_'+str(y)+' '
            pt_str += pt

        keypoint_word.append(pt_str)
        x0_, y0_, x1_, y1_ = bbox
        x0.append(x0_)
        y0.append(y0_)
        x1.append(x1_)
        y1.append(y1_)

    pd.DataFrame({'Image':img_id, 'x0':x0, 'y0':y0, 'x1':x1, 'y1':y1}).to_csv( '{}.csv'.format(name+'_bbox'), index=None)

