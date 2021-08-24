import sys
sys.path.append('..')

import os
import json
import time
import math
import argparse


import cv2
import numpy as np


import paddle
import pose_estimation


from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

#from skimage.feature import peak_local_max

limbSeq = [[1,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8], [2,15], [15,12], [12,13], [13,14], [15,9], [9,10], [10, 11]]

mapIdx = [[16,17],[18,19],[20,21],[22,23],[24,25],[26,27],[28,29],[30,31],[38,39],[40,41],[42,43], \
          [32,33],[34,35],[36,37]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255]]

orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 6]
targetDist = 41/35
boxsize = 368
scale_search = [0.7, 1, 1.3]
stride = 8
padValue = 128
thre_point = 0.15
thre_line = 0.05
stickwidth = 7



def construct_model(args):

    model = pose_estimation.PoseModel(num_point=16, num_vector=14)
    state_dict = paddle.load(args.model)['state_dict']
    model.set_state_dict(state_dict)
    model.eval()

    return model

def padRightDownCorner(img, stride, padValue):

    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def normalize(origin_img):
    origin_img = np.array(origin_img, dtype=np.float32)
    origin_img -= 128.0
    origin_img /= 256.0

    return origin_img



def eval_mpii(preds):


    threshold = 0.5
    SC_BIAS = 0.6
    dict = loadmat('../data/mpii/annot/gt_valid.mat')
    dataset_joints = dict['dataset_joints']
    jnt_missing = dict['jnt_missing']#[:, :20]
    pos_pred_src = dict['pos_pred_src']#[:, :, :20]
    pos_gt_src = dict['pos_gt_src']#[:, :, :20]
    headboxes_src = dict['headboxes_src']#[:, :, :20]
    #predictions
    pos_pred_src = np.transpose(preds, [1, 2, 0])
    

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)


    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)


    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    print('\nPrediction file: {}\n'.format('File'))
    print("Head,   Shoulder, Elbow,  Wrist,   Hip ,     Knee  , Ankle ,  Mean")
    print('{:.2f}  {:.2f}     {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
            , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
            , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))

def process(model, image, meta, index1):

    origin_img = image
    #normed_img = normalize(origin_img)

    height, width, _ = origin_img.shape
    pos = np.array([meta['center']])
    scale = np.array(meta['scale'])

    minX = np.min(pos[:,0])
    minY = np.min(pos[:,1])
    maxX = np.max(pos[:,0])
    maxY = np.max(pos[:,1])

    zeroScaleIdx = scale != 0
    scale = scale[zeroScaleIdx]

    if np.sum(zeroScaleIdx) ==0:
        scale = [1]
    scale0 = targetDist / np.mean(scale)
    deltaX = boxsize / (scale0 * 2.5)
    deltaY = boxsize / (scale0 * 2)

    bbox = np.zeros((4,))
    dX = deltaX * 0.25
    dY = deltaY * 0.25

    bbox[0] = int(np.round(np.max(minX-dX,0)))
    bbox[1] = int(np.round(np.max(minY-dY,0)))
    bbox[2] = int(np.round(np.min([maxX+dX,np.shape(origin_img)[1]])))
    bbox[3] = int(np.round(np.min([maxY+dY,np.shape(origin_img)[0]])))
    bbox = np.array(bbox, dtype=int)

    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 16)) # num_point
    paf_avg = np.zeros((height, width, 28))     # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(origin_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, stride, padValue)

        input_img = np.transpose(imgToTest_padded[:,:,:,np.newaxis], (3, 2, 0, 1)) # required shape (1, c, h, w)
        input_img = np.float32(input_img / 256 - 0.5)
        input_var = paddle.to_tensor(input_img, stop_gradient=False)

        # get the features
        vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var)

        # get the heatmap
        heatmap = heat6.numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0)) # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier) #final_score

        # get the paf
        paf = vec6.numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0)) # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)
    
    #print(heatmap_avg)

    # non-maximum suppression for finding joint candidates
    all_peaks = []   # all of the possible points by classes.
    peak_counter = 0
    for part in range(0, 16):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # (w, h)
        
        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i], ) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    
    connection_all = [] # save all of the possible lines by classes.
    special_k = []      # save the lines, which haven't legal points.
    mid_num = 10        # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x - 16 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2]) # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1, 0) # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3], reverse=True) # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]]) 
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])


    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 3 or subset[i][-2] / subset[i][-1] < 0.1:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    point_cnt = 0
    for ridxPred in range(len(subset)):
        point = np.zeros((16, 2))
        sum_x = 0
        sum_y = 0
        part_cnt = 0
        for part in range(14):
            index = subset[ridxPred, part]
            if index > 0:
                x = candidate[index.astype('int'), 0]
                y = candidate[index.astype('int'), 1]
                idx = orderMPI[part]
                point[idx] = [x, y]
                sum_x = sum_x + x
                sum_y = sum_y + y
                part_cnt = part_cnt + 1
        mean_x = sum_x / part_cnt
        mean_y = sum_y / part_cnt
        index = int(subset[ridxPred, 14])
        if mean_x > bbox[0] and mean_x < bbox[2] and mean_y > bbox[1] and mean_y < bbox[3]:
            preds[index1] = point #point
            point_cnt = point_cnt + 1
        elif index > 0 and candidate[index, 0] > bbox[0] and candidate[index, 0] < bbox[2] and candidate[index, 1] > bbox[1] and candidate[index, 1] < bbox[3]:
            preds[index1] = point #point
            point_cnt = point_cnt + 1
        else:
            pass
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pose_iter_rename.pdparams', help='path to the weights file')
    parser.add_argument('-only_eval', type=bool, help='./output/predection.npy exists')

    args = parser.parse_args()


    # load model
    model = construct_model(args)

    tic = time.time()
    print('start processing...')

    with open('../data/mpii/annot/valid.json') as f:
        pred = json.load(f)
    pred = pred
    
    if args.only_eval:
        preds = np.load('./output/predection.npy')
    else:
        images = {}
        for i in tqdm(range(len(pred))):
            images[pred[i]['image']] = cv2.imread('../data/mpii/images/' + pred[i]['image'])
        # generate image with body parts
        preds = np.zeros((len(pred), 16, 2))
        for i in tqdm(range(len(pred))):
            process(model, images[pred[i]['image']], pred[i], i)
        np.save("./output/predection.npy",preds)
    eval_mpii(preds)
    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))
