"""CPM Pytorch Implementation"""

from collections import OrderedDict

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import initializer as init

class ReLU_(paddle.nn.ReLU):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.nn.functional.relu_(x)
        return out

def constant_init_(param, val):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.full(param.shape, val, param.dtype)))
    paddle.assign(replaced_param, param)


def normal_init_(param, mean=0.0, std=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.normal(
                mean=mean, std=std, shape=param.shape)))
    paddle.assign(replaced_param, param)


def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2D(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2D(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, ReLU_()]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2D(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2D(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2D(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, ReLU_()]
    return nn.Sequential(*layers)



def get_model(trunk='vgg19'):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models['block0'] = make_vgg19_block(block0)

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    class rtpose_model(nn.Layer):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            out1 = self.model0(x)
            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = paddle.concat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = paddle.concat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = paddle.concat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = paddle.concat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = paddle.concat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss

        def _initialize_weights_norm(self):

            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    v = np.random.normal(scale = 0.01, size=m.weight.shape).astype('float32')
                    m.weight.set_value(v)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        m.bias.set_value(np.ones(m.bias.shape).astype('float32'))

            # last layer of these block don't have Relu
            #init.Normal(self.model1_1[8].weight, )
            self.model1_1[8].weight.set_value(np.random.normal(scale=0.01, size=self.model1_1[8].weight.shape).astype('float32'))
            self.model1_2[8].weight.set_value(np.random.normal(scale=0.01, size=self.model1_2[8].weight.shape).astype('float32'))
            
            self.model2_1[12].weight.set_value(np.random.normal(scale=0.01, size=self.model2_1[12].weight.shape).astype('float32'))
            self.model3_1[12].weight.set_value(np.random.normal(scale=0.01, size=self.model3_1[12].weight.shape).astype('float32'))
            self.model4_1[12].weight.set_value(np.random.normal(scale=0.01, size=self.model4_1[12].weight.shape).astype('float32'))
            self.model5_1[12].weight.set_value(np.random.normal(scale=0.01, size=self.model5_1[12].weight.shape).astype('float32'))
            self.model6_1[12].weight.set_value(np.random.normal(scale=0.01, size=self.model6_1[12].weight.shape).astype('float32'))

            self.model2_2[12].weight.set_value(np.random.normal(scale=0.01, size=self.model2_2[12].weight.shape).astype('float32'))
            self.model3_2[12].weight.set_value(np.random.normal(scale=0.01, size=self.model3_2[12].weight.shape).astype('float32'))
            self.model4_2[12].weight.set_value(np.random.normal(scale=0.01, size=self.model4_2[12].weight.shape).astype('float32'))
            self.model5_2[12].weight.set_value(np.random.normal(scale=0.01, size=self.model5_2[12].weight.shape).astype('float32'))
            self.model6_2[12].weight.set_value(np.random.normal(scale=0.01, size=self.model6_2[12].weight.shape).astype('float32'))

    model = rtpose_model(models)
    return model


"""Load pretrained model on Imagenet
:param model, the PyTorch nn.Layer which will train.
:param model_path, the directory which load the pretrained model, will download one if not have.
:param trunk, the feature extractor network of model.               
"""


def use_vgg(model):

    #url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    vgg_state_dict = paddle.load('VGG19_pretrained.pdparams')
    vgg_keys = vgg_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(20):
        weights_load[list(model.state_dict().keys())[i]
                     ] = vgg_state_dict[list(vgg_keys)[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.set_state_dict(state)
    print('load imagenet pretrained model')