import paddle
import paddle.io as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
from PIL import Image
import cv2
import Mytransforms

def read_data_file(file_dir):

    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists

def read_json_file(file_dir):
    """
        filename: JSON file
        return: two list: key_points list and centers list
    """
    fp = open(file_dir)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []
    img_list = []
    for info in data:
        img_list.append(info['image'])
        kpt = []
        center = []
        scale = []
        lists = info['info']
        for x in lists:
           kpt.append(x['joints'])
           center.append(x['center'])
           scale.append(x['scale'])
        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
    fp.close()

    return img_list, kpts, centers, scales

def generate_heatmap(heatmap, kpt, stride, sigma):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5
    num = len(kpt)
    length = len(kpt[0]) - 1
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] > 1:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j + 1] += math.exp(-dis)
                    if heatmap[h][w][j + 1] > 1:
                        heatmap[h][w][j + 1] = 1

    return heatmap

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector

class MPIIFolder(data.Dataset):

    def __init__(self, file_dir, stride, transformer=None):

        self.img_list, self.kpt_list, self.center_list, self.scale_list = read_json_file(file_dir)
        self.stride = stride
        self.transformer = transformer
        self.vec_pair = [[2,3,5,6,8,9, 11,12,0,1,1, 1,1, 0],
                         [3,4,6,7,9,10,12,13,1,8,11,2,5, 14]] # different from openpose

        
        self.theta = 1.0
        self.sigma = 7.0
        
    def __getitem__(self, index):
        idx = index
        img_path = self.img_list[index]
        img = np.array(cv2.imread('../data/mpii/images/' + img_path), dtype=np.float32)
        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        img, kpt, center = self.transformer(img, kpt, center, scale)

        height, width, _ = img.shape
        heatmap = np.zeros((int(height / self.stride), int(width / self.stride), len(kpt[0])), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, kpt, self.stride, self.sigma)
        heatmap[:,:,0] = 1.0 - np.max(heatmap[:,:,1:], axis=2) # for background
        heatmap = heatmap

        vecmap = np.zeros((int(height / self.stride), int(width / self.stride), len(self.vec_pair[0]) * 2), dtype=np.float32)
        cnt = np.zeros((int(height / self.stride), int(width / self.stride), len(self.vec_pair[0])), dtype=np.int32)

        vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
        vecmap = vecmap

        normalize = paddle.vision.transforms.Normalize(mean=[128.0, 128.0, 128.0], std = [256.0, 256.0, 256.0])
        img = normalize(Mytransforms.to_tensor(img)) # mean, std
        heatmap = Mytransforms.to_tensor(heatmap)
        vecmap = Mytransforms.to_tensor(vecmap)

        return img, heatmap, vecmap

    def __len__(self):

        return len(self.img_list)