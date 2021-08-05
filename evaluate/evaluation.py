import unittest
import paddle
from collections import OrderedDict
#from evaluate.coco_eval import run_eval
from coco_eval import run_eval
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.network.openpose import OpenPose_Model, use_vgg

#Notice, if you using the 
with paddle.no_grad():
    # this path is with respect to the root of the project
    weight_name = 'pose_model.pdparams'
    state_dict = paddle.load(weight_name)
    
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[6:]
        new_state_dict[name]=v
        
    model = get_model(trunk='vgg19')
    #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    #model = torch.nn.DataParallel(model).cuda()
    model.set_state_dict(new_state_dict)
    model.eval()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    run_eval(image_dir= '/data/coco/images/val2017', anno_file = '/data/coco/annotations/person_keypoints_val2017.json', vis_dir = '/data/coco/images/vis_val2017', model=model, preprocess='vgg')
