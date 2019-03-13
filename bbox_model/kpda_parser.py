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

