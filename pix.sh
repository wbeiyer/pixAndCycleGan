#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode aligned --direction AtoB

# python train.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode template > output.log 2>&1
# python test.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode template > output1.log 2>&1