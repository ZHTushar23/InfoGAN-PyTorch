#!/bin/bash
device=$1
export CUDA_LAUNCH_BLOCKING=1
python v18_train_cloud.py --device $device