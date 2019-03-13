import math
import numpy as np

class Config:

    def __init__(self, clothes):

        self.batch_size = 8
        self.workers = 16
        self.base_lr = 1e-2  # learning rate
        self.epochs = 120

        self.clothes = clothes

        self.keypoints = {'whale' : [
            'left_fluke_tip',
            'left_notch',
            'center_notch',
            'right_notch',
            'right_fluke_tip',
            'right_fluke_leading',
            'right_peduncle_or_water',
            'center_peduncle_or_water',
            'left_peduncle_or_water',
            'left_fluke_leading',],}

        keypoint = self.keypoints[self.clothes]
        self.num_keypoints = len(keypoint)
        self.conjug = []

        for i, key in enumerate(keypoint):
            print(key)
            if 'left' in key:
                print(key)
                j = keypoint.index(key.replace('left', 'right'))
                self.conjug.append([i, j])

        # print(self.conjug)

        # Img
        self.img_max_size = 512
        self.mu = 0.65
        self.sigma = 0.25

        # STAGE 2
        self.hm_stride = 4
        # Heatmap

        self.hm_sigma = self.img_max_size / self.hm_stride / 16. #4 #16 for 256 size
        self.hm_alpha = 100.

        lrschedule = {'whale' : [30, 60, 90]}
        self.lrschedule = lrschedule[clothes]


if __name__ == '__main__':
    config = Config('whale')
    print(config.conjug)