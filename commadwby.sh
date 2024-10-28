#! /bin/bash
### 表示这是一个bash脚本

#SBATCH --job-name=wbyPix
### 设置该作业的作业名

#SBATCH --nodes=1
### 指定该作业需要1个节点数

#SBATCH --ntasks-per-node=4
### 每个节点所运行的进程数为4

#SBATCH --time=24:00:00
### 作业最大的运行时间，超过时间后作业资源会被SLURM回收

#SBATCH --comment jbgs_grp
### 指定从哪个项目扣费。如果没有这条参数，则从个人账户扣费

#SBATCH --mail-type=end
#SBATCH --mail-user=905906821@qq.com

#SBATCH --ntasks=4
### 该作业需要4个CPU

#SBATCH --gres=gpu:1
#SBATCH --partition=g078t2
### 申请1块GPU卡
source ~/.bashrc
export TMPDIR=$HOME/tmp

### 初始化环境变量
conda activate bright
cd /home/u2022111265/code/pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode aligned --direction AtoB > outputPixTrain.log 2>&1
# python train.py --dataroot ./datasets/thread --name thread_cyclegan --model cycle_gan --dataset_mode unaligned > outputClcTrain.log 2>&1
# python train.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode template > output.log 2>&1
# python test.py --dataroot ./datasets/thread --name thread_pix2pix --model pix2pix --dataset_mode template > output1.log 2>&1
### 程序的执行命令