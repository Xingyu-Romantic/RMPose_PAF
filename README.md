# RMPose_PAF
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ∗ 论文复现

## Train

``` python
export PYTHONPATH="$PWD":$PYTHONPATH
python training/train_pose.py --config ./trainning/config.yml --train_dir ./process_train.json --val_dir ./process_val.json



