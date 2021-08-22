# RMPose_PAF

Ai Studio: https://aistudio.baidu.com/aistudio/projectdetail/2267186?shared=1

《Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ∗ 》论文复现

训练权重： https://drive.google.com/file/d/1YOj_GE07bvUlAGkdgM1Q8blsZvzvMTNh/view?usp=sharing

下载后放在当前目录

## Structure

`pose_estimation.py`:   `Model `

`ski.jpg` : Test image

`nohup.out`: Train Log

## Train

``` python
export PYTHONPATH="$PWD":$PYTHONPATH
python training/train_pose.py --config ./trainning/config.yml --train_dir ./process_train.json --val_dir ./process_val.json
```

## Test

已经将测试集的图片预测结果保存为`predection.npy`

若要重新评估，取消426~429行注释， 注释431行

```python
python testing/eval_new.py
```

![](https://github.com/Xingyu-Romantic/RMPose_PAF/blob/main/result.png)

## Inference

```python
python testing/test_pose.py --image ski.jpg
```

## Question

Q: **ModuleNotFoundError**: No module named 'pafprocess_mpi'

A: `export PYTHONPATH="$PWD":$PYTHONPATH`

