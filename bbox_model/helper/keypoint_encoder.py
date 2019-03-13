import torch
import numpy as np
import torch.nn.functional as F
import cv2

class KeypointEncoder:

    def _gaussian_keypoint(self, input_size, mu_x, mu_y, alpha, sigma):

        mu_x = mu_x.float()
        mu_y = mu_y.float()

        h, w = input_size
        x = torch.linspace(0, w-1, steps=w)
        y = torch.linspace(0, h-1, steps=h)
        xx = x.repeat(w, 1).float()
        yy = y.repeat(h, 1).t().float()

        tmp = -( ((xx-mu_x)**2) / (2*sigma**2) + ((yy-mu_y)**2) / (2*sigma**2) )
        zz = alpha * torch.exp(tmp)
        return zz

    def _gaussian_keypoint_np(self, input_size, mu_x, mu_y, alpha, sigma):

        mu_x = mu_x.float()
        mu_y = mu_y.float()

        h, w = input_size
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        xx, yy = np.meshgrid(x, y)
        zz = alpha * np.exp(-( ((xx-mu_x)**2) / (2*sigma**2) + ((yy-mu_y)**2) / (2*sigma**2) ))
        return zz

    def encode(self, keypoints, input_size, stride, hm_alpha, hm_sigma):
        '''
        :param keypoints: [pt num, 3] -> [[x, y, vis], ...]
        :return: [pt num, h, w] [h, w]
        '''
        kpt_num = len(keypoints)
        vismap = torch.zeros([kpt_num, ])
        inps = [x//stride for x in input_size]
        kpts = keypoints.clone()
        kpts[:,:2] = (kpts[:,:2] - 1.0) / stride
        h, w = inps
        heatmap = torch.zeros([kpt_num, h, w])
        for c, kpt in enumerate(kpts):
            x, y, v = kpt
            if v >= 0:
            #if v == 1 and x > 0 and y > 0:
                heatmap[c] = self._gaussian_keypoint(inps, x, y, hm_alpha, hm_sigma)
            vismap[c] = v
        return heatmap, vismap

    def decode_np(self, heatmap, scale, stride, default_pt, method='exp'):
        '''
        :param heatmap: [pt_num, h, w]
        :param scale:
        :return: 
        '''

        kp_num, h, w = heatmap.shape
        dfx, dfy = np.array(default_pt) * scale / (stride*1.0)
        for k, hm in enumerate(heatmap):
            heatmap[k] = cv2.GaussianBlur(hm, (5, 5), 1)
        if method == 'exp':
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            heatmap_th = np.copy(heatmap)
            heatmap_th[heatmap<np.amax(heatmap)/2.0] = 0
            heat_sums_th = np.sum(heatmap_th, axis=(1, 2))
            x = np.sum(heatmap_th * xx, axis=(1, 2))
            y = np.sum(heatmap_th * yy, axis=(1, 2))
            x = x / heat_sums_th
            y = y / heat_sums_th
            x[heat_sums_th == 0] = dfx
            y[heat_sums_th == 0] = dfy
        else:
            if method == 'max':
                heatmap_th = heatmap.reshape(kp_num, -1)
                y, x = np.unravel_index(np.argmax(heatmap_th, axis=1), [h, w])
            elif method == 'maxoffset':
                heatmap_th = heatmap.reshape(kp_num, -1)
                si = np.argsort(heatmap_th, axis=1)
                y1, x1 = np.unravel_index(si[:, -1], [h, w])
                y2, x2 = np.unravel_index(si[:, -2], [h, w])
                x = (3 * x1 + x2) / 4.
                y = (3 * y1 + y2) / 4.
            var = np.var(heatmap_th, axis=1)
            x[var<1] = dfx
            y[var<1] = dfy
        x = x * stride / (scale*1.0)
        y = y * stride / (scale*1.0)
        return np.rint(x+2), np.rint(y+2)



if __name__ == '__main__':
    from bbox_model.kpda_parser import KPDA
    import cv2
    from bbox_model.config import Config
    import numpy as np
    config = Config('whale')
    # db_path = '/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/train/'
    kpda = KPDA(config)
    img_path = kpda.get_image_path(0)
    kpts = kpda.get_keypoints(0)
    kpts = torch.from_numpy(kpts)
    img = cv2.imread(img_path,1)
    image = np.zeros([2048, 2048, 3])
    image[:img.shape[0], :img.shape[1], :] = img

    cv2.imwrite('img.jpg', image)
    ke = KeypointEncoder()
    heatmaps, _ = ke.encode(kpts, image.shape[:2], config.hm_stride, config.hm_sigma, config.hm_sigma)
    for i, heatmap in enumerate(heatmaps):
        heatmap = np.expand_dims(heatmap.numpy() * 255, 2)
        cv2.imwrite('map%d.jpg' % i, heatmap )