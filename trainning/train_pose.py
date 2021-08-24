import paddle
import paddle.nn as nn
import os
import sys
import argparse
import time
sys.path.append('..')
import MPIIFolder
import Mytransforms
from utils import AverageMeter as AverageMeter
from utils import save_checkpoint as save_checkpoint
from utils import Config as Config
import pose_estimation
import numpy as np

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='./openpose_coco_latest.pdparams.tar',type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default='../data/mpii/images/', type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')

    return parser.parse_args()


def construct_model(args):

    model = pose_estimation.PoseModel(num_point=16, num_vector=14, pretrained=False)
    state_dict = paddle.load(args.pretrained)['state_dict']
    model.set_state_dict(state_dict)
    model = paddle.DataParallel(model)

    return model



def train_val(model, args):

    traindir = args.train_dir
    valdir = args.val_dir

    config = Config(args.config)
    
    train_loader = paddle.io.DataLoader(
            MPIIFolder.MPIIFolder(traindir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers)

    if config.test_interval != 0 and args.val_dir is not None:
        val_loader = paddle.io.DataLoader(
                MPIIFolder.MPIIFolder(valdir, 8,
                    Mytransforms.Compose([Mytransforms.TestResized(368),
                ])),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers)
    
    criterion = nn.MSELoss()

    
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate = config.base_lr, step_size = config.step_size, gamma = config.gamma)
    optimizer = paddle.optimizer.Momentum(scheduler, momentum=config.momentum, parameters = model.parameters(),
                                weight_decay=config.weight_decay)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(12)]
    
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model
    learning_rate = config.base_lr

    model.train()

    heat_weight = 46 * 46 * 16 / 2.0 # for convenient to compare with origin code
    vec_weight = 46 * 46 * 28 / 2.0

    while iters < config.max_iter:
    
        for i, (input, heatmap, vecmap) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input_var = paddle.to_tensor(np.array(input), stop_gradient=False)
            heatmap_var = paddle.to_tensor(np.array(heatmap), stop_gradient=False)
            vecmap_var = paddle.to_tensor(np.array(vecmap), stop_gradient=False)

            vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var)
            loss1_1 = criterion(vec1, vecmap_var) * vec_weight
            loss1_2 = criterion(heat1, heatmap_var) * heat_weight
            loss2_1 = criterion(vec2, vecmap_var) * vec_weight
            loss2_2 = criterion(heat2, heatmap_var) * heat_weight
            loss3_1 = criterion(vec3, vecmap_var) * vec_weight
            loss3_2 = criterion(heat3, heatmap_var) * heat_weight
            loss4_1 = criterion(vec4, vecmap_var) * vec_weight
            loss4_2 = criterion(heat4, heatmap_var) * heat_weight
            loss5_1 = criterion(vec5, vecmap_var) * vec_weight
            loss5_2 = criterion(heat5, heatmap_var) * heat_weight
            loss6_1 = criterion(vec6, vecmap_var) * vec_weight
            loss6_2 = criterion(heat6, heatmap_var) * heat_weight
            
            loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2

            losses.update(loss[0].item(), input.size)
            for cnt, l in enumerate([loss1_1, loss1_2, loss2_1, loss2_2, loss3_1, loss3_2, loss4_1, loss4_2, loss5_1, loss5_2, loss6_1, loss6_2]):
                losses_list[cnt].update(l[0].item(), input.size)
            
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0,12,2):
                    print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                        'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1, loss1=losses_list[cnt], loss2=losses_list[cnt + 1]))
                print(time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(12):
                    losses_list[cnt].reset()
    
            if config.test_interval != 0 and args.val_dir is not None and iters % config.test_interval == 0:

                with paddle.no_grad():
                    model.eval()
                    for j, (input, heatmap, vecmap) in enumerate(val_loader):

                        input_var = paddle.to_tensor(np.array(input), stop_gradient=False)
                        heatmap_var = paddle.to_tensor(np.array(heatmap), stop_gradient=False)
                        vecmap_var = paddle.to_tensor(np.array(vecmap), stop_gradient=False)

                        vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var)
                        loss1_1 = criterion(vec1, vecmap_var) * vec_weight
                        loss1_2 = criterion(heat1, heatmap_var) * heat_weight
                        loss2_1 = criterion(vec2, vecmap_var) * vec_weight
                        loss2_2 = criterion(heat2, heatmap_var) * heat_weight
                        loss3_1 = criterion(vec3, vecmap_var) * vec_weight
                        loss3_2 = criterion(heat3, heatmap_var) * heat_weight
                        loss4_1 = criterion(vec4, vecmap_var) * vec_weight
                        loss4_2 = criterion(heat4, heatmap_var) * heat_weight
                        loss5_1 = criterion(vec5, vecmap_var) * vec_weight
                        loss5_2 = criterion(heat5, heatmap_var) * heat_weight
                        loss6_1 = criterion(vec6, vecmap_var) * vec_weight
                        loss6_2 = criterion(heat6, heatmap_var) * heat_weight
                        
                        loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2

                        losses.update(loss[0].item(), input.size)
                        for cnt, l in enumerate([loss1_1, loss1_2, loss2_1, loss2_2, loss3_1, loss3_2, loss4_1, loss4_2, loss5_1, loss5_2, loss6_1, loss6_2]):
                            losses_list[cnt].update(l[0].item(), input.size)
        
                    batch_time.update(time.time() - end)
                    end = time.time()
                    is_best = losses.avg < best_model
                    best_model = min(best_model, losses.avg)
                    save_checkpoint({
                        'iter': iters,
                        'state_dict': model.state_dict(),
                        }, is_best, 'openpose_mpii')
        
                    print(
                        'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                        'Loss {loss.avg:.8f}\n'.format(
                        batch_time=batch_time, loss=losses))
                    for cnt in range(0,12,2):
                        print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                            'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1, loss1=losses_list[cnt], loss2=losses_list[cnt + 1]))
                    print(time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime()))
        
                    batch_time.reset()
                    losses.reset()
                    for cnt in range(12):
                        losses_list[cnt].reset()
                
            model.train()
            scheduler.step()
            if iters == config.max_iter:
                break


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    paddle.set_device('gpu')
    args = parse()
    model = construct_model(args)
    train_val(model, args)