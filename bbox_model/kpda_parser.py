import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

import os
TRN_IMGS_DIR = '/data1/shentao/DATA/competitions/whale/train/'
TST_IMGS_DIR = '/data1/shentao/DATA/competitions/whale/test/'

class KPDA():
    def __init__(self, config, mode = 'train'):
        self.config = config
        self.mode = mode

        if mode == 'infer':
            list_train = os.listdir(TRN_IMGS_DIR)
            list_test = os.listdir(TST_IMGS_DIR)
            self.anno_df = list_train + list_test
            self.anno_df = [tmp for tmp in self.anno_df if '.jpg' in tmp]

    def size(self):
        return len(self.anno_df)

    def get_image_path(self, image_index):
        image_path = self.anno_df[image_index]
        return image_path

    def get_bbox(self, image_index, extend=10):
        row = self.anno_df[image_index]
        locs = []

        for key, item in row.iteritems():
            locs.append(item)

        locs = locs[1:5]
        locs_final = []

        locs_final.append(np.asarray([int(locs[2 * 0]), int(locs[2 * 0 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 0]), int(locs[2 * 1 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 1]), int(locs[2 * 0 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 1]), int(locs[2 * 1 + 1]),1]))


        locs = np.array(locs_final)
        minimums = np.amin(locs, axis=0)
        maximums = np.amax(locs, axis=0)
        bbox = np.array([[max(minimums[0]-extend, 0), max(minimums[1]-extend, 0),
                          maximums[0]+extend, maximums[1]+extend]], dtype=np.float32)
        return bbox

    def get_keypoints(self, image_index):
        row = self.anno_df[image_index]
        locs = []

        for key, item in row.iteritems():
            locs.append(item)

        locs = locs[1:1+4]
        locs_final = []
        locs_final.append(np.asarray([int(locs[2 * 0]), int(locs[2 * 0 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 0]), int(locs[2 * 1 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 1]), int(locs[2 * 0 + 1]),1]))
        locs_final.append(np.asarray([int(locs[2 * 1]), int(locs[2 * 1 + 1]),1]))

        locs = np.array(locs_final)
        return locs

# def get_default_xy(config, db_path):
#     kpda = KPDA(config)
#     s = []
#     for k in range(kpda.size()):
#         path = kpda.get_image_path(k)
#         img = cv2.imread(path)
#         h, w, _ = img.shape
#         locs = kpda.get_keypoints(k).astype(np.float32)
#         locs[:, 0] = locs[:, 0].astype(np.float32) / float(w)
#         locs[:, 1] = locs[:, 1].astype(np.float32) / float(h)
#         locs[:, 2] = (locs[:, 2]>=0)*1.0
#         s.append(locs)
#     s = np.stack(s)
#     s = s.sum(axis=0)
#     s = s[:, :2] / s[:, 2:].repeat(2, axis=1)
#     print(s)

# if __name__ == '__main__':
#     from tqdm import tqdm
#     import cv2
#     from config import Config
#     config = Config('whale')
#     kpda = KPDA(config)
#     pts = kpda.get_keypoints(100)
#     bbox = kpda.get_bbox(100)
#     img_path = kpda.get_image_path(100)
#     print(img_path)
#     print(pts)
#     print(bbox)

