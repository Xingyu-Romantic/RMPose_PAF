import os
import json

import numpy as np
from scipy.io import loadmat

def pre(annot_file):
    with open(annot_file) as f:
        annot = json.load(f)
    '''
    data: [{'image', 'info': {}, {}}, {}]

    '''
    
    images = set()
    for i in annot:
        images.add(i['image'])
    images = list(images)
    gt_valid_multi = []
    for i in range(len(images)):
        gt_valid_multi.append({'image': images[i], 'num_people': 0})

    for i in annot:
        for j in range(len(gt_valid_multi)):
            if i['image'] == gt_valid_multi[j]['image']:
                gt_valid_multi[j]['num_people'] = gt_valid_multi[j].get('num_people', 0) + 1
                gt_valid_multi[j]['num_people'] = gt_valid_multi[j].get('num_people', 0) + 1
                gt_valid_multi[j]['keypoint_multi'] = gt_valid_multi[j].get('keypoint_multi', np.zeros((16, 2))) + i['joints']
                gt_valid_multi[j]['']
                #vis = list(np.array(i['joints_vis']).reshape(16,1))
                #key = np.hstack((keypoint, vis))
                #assert key.shape == (16, 3)
                #j['info'].append({'scale':i['scale'], 'center': i['center'], 'joints':key.tolist()})
    for j in gt_valid_multi:
        j['keypoint_multi'] = j['keypoint_multi'].tolist()
    with open( './datasets/process_' + annot_file.split('/')[-1], 'w') as f:
        json.dump(gt_valid_multi, f)
    

pre('../data/mpii/annot/valid.json')
pre('../data/mpii/annot/trainval.json')
pre('../data/mpii/annot/valid.json')