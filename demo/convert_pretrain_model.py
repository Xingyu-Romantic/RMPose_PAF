from collections import OrderedDict
import torch

def load_pytorch_pretrain_model(paddle_model, path):
    '''
    paddle_model: dygraph layer object
    pytorch_state_dict: pytorch state_dict, assume in CPU device
    '''

    paddle_weight = paddle_model.state_dict()
    torch_weight = torch.load(path)

    torch_keys = []  # 存放torch模型的权重键值
    paddle_keys = []  # 存放paddle模型的权重键值

    for k in torch_weight:  # 遍历torch模型权重键值
        torch_keys.append(k)

    for k in paddle_weight:  # 遍历paddle模型权重键值
        paddle_keys.append(k)

    key_pair_length = min(len(torch_keys), len(paddle_keys)) # 获取最小对应权重长度

    # 将pytorch模型参数赋值给paddle模型
    for i, k in enumerate(paddle_keys):
        if i >= key_pair_length:
            break
        if paddle_weight[k].shape == torch_weight[torch_keys[i]].detach().numpy().shape: # 权重参数shape比较，只有一一对应才会赋值
            paddle_weight[k] = torch_weight[torch_keys[i]].detach().numpy()

    # 将paddle模型参数赋值给pytorch模型
    for i, k in enumerate(torch_keys):
        if i >= key_pair_length:
            break
        if torch_weight[k].detach().numpy().shape == paddle_weight[paddle_keys[i]].shape: # 权重参数shape比较，只有一一对应才会赋值
            torch_weight[k] = paddle_weight[paddle_keys[i]]
    return paddle_model.state_dict()
