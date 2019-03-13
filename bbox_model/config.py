import math
import numpy as np

class Config:

    def __init__(self, type):
        self.batch_size = 8 * 16
        self.workers = 16
        self.base_lr = 1e-2  # learning rate
        self.epochs = 50

        self.type = type
        self.keypoints = {'whale':
                        ['left_down_point',
                         'left_up_point',
                         'right_down_point',
                         'right_up_point']}

        keypoint = self.keypoints[self.type]
        self.num_keypoints = len(keypoint)
        self.conjug = []

        for i, key in enumerate(keypoint):
            print(key)
            if 'left' in key:
                print(key)
                j = keypoint.index(key.replace('left', 'right'))
                self.conjug.append([i, j])

        # Img
        self.img_max_size = 256
        self.mu = 0.65
        self.sigma = 0.25

        # STAGE 2
        self.hm_stride = 4
        # Heatmap

        self.hm_sigma = self.img_max_size / self.hm_stride / 16. #4 #16 for 256 size
        self.hm_alpha = 100.

        lrschedule = {'whale' : [4, 8, 12]}
        self.lrschedule = lrschedule[type]


if __name__ == '__main__':
    config = Config('whale')
    print(config.conjug)