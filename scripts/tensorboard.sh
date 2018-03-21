#!/bin/bash

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

log_dir=/zfs/zhang/TrainLog/logs/MobileNetV2Share-2018-03-21-08-48-03/
tb_dir=/afs/cg.cs.tu-bs.de/home/zhang/.local/lib/python3.5/site-packages/tensorboard/
python ${tb_dir}main.py --logdir=${log_dir} --port=8008