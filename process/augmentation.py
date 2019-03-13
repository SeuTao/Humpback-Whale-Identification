import imgaug.augmenters as iaa
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import os
import numpy as np
import random
import skimage
#===================================================paug===============================================================
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    original = np.array([[0, 0],
                         [image.shape[1] - 1, 0],
                         [image.shape[1] - 1, image.shape[0] - 1],
                         [0, image.shape[0] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(original, rect)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped

def Perspective_aug(img,   threshold1 = 0.25, threshold2 = 0.75):
    # img = cv2.imread(img_name)
    rows, cols, ch = img.shape

    x0,y0 = random.randint(0, int(cols * threshold1)), random.randint(0, int(rows * threshold1))
    x1,y1 = random.randint(int(cols * threshold2), cols - 1), random.randint(0, int(rows * threshold1))
    x2,y2 = random.randint(int(cols * threshold2), cols - 1), random.randint(int(rows * threshold2), rows - 1)
    x3,y3 = random.randint(0, int(cols * threshold1)), random.randint(int(rows * threshold2), rows - 1)
    pts = np.float32([(x0,y0),
                      (x1,y1),
                      (x2,y2),
                      (x3,y3)])

    warped = four_point_transform(img, pts)

    x_ = np.asarray([x0, x1, x2, x3])
    y_ = np.asarray([y0, y1, y2, y3])

    min_x = np.min(x_)
    max_x = np.max(x_)
    min_y = np.min(y_)
    max_y = np.max(y_)

    warped = warped[min_y:max_y,min_x:max_x,:]
    return warped

#===================================================origin=============================================================
def aug_image(image, is_infer=False, augment = None):
    if is_infer:
        flip_code = augment[0]

        if flip_code == 1:
            seq = iaa.Sequential([iaa.Fliplr(1.0)])
        elif flip_code == 2:
            seq = iaa.Sequential([iaa.Flipud(1.0)])
        elif flip_code == 3:
            seq = iaa.Sequential([iaa.Flipud(1.0),
                                  iaa.Fliplr(1.0)])
        elif flip_code ==0:
            return image

    else:

        seq = iaa.Sequential([
            iaa.Affine(rotate= (-15, 15),
                       shear = (-15, 15),
                       mode='edge'),

            iaa.SomeOf((0, 2),
                       [
                           iaa.GaussianBlur((0, 1.5)),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                           iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                           iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                           iaa.PerspectiveTransform(scale=(0.01, 0.1))
                       ],
                       random_order=True
                       )
        ])

    image = seq.augment_image(image)
    return image

#===================================================crop===============================================================
def get_cropped_img(image, bbox, is_mask=False):
    crop_margin = 0.1

    size_x = image.shape[1]
    size_y = image.shape[0]

    x0, y0, x1, y1 = bbox

    dx = x1 - x0
    dy = y1 - y0

    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1

    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y

    if is_mask:
        crop = image[int(y0):int(y1), int(x0):int(x1)]
    else:
        crop = image[int(y0):int(y1), int(x0):int(x1), :]

    return crop

def get_center_aligned_img(crop_img, pt):
    x0, y0 = pt[2]
    x1, y1 = pt[7]

    x = int((x0 + x1) / 2.0)
    w = crop_img.shape[1]
    radius = max(x, w - x)
    s = radius - x
    fill = np.ones([crop_img.shape[0], radius * 2 + 1, 3]).astype(np.uint8) * 128

    if x < radius:
        fill[:, s:s + crop_img.shape[1], :] = crop_img
    else:
        fill[:, 0:crop_img.shape[1], :] = crop_img

    return fill
