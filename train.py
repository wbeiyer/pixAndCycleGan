"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

from options.train_options import TrainOptions 
from options.val_options import ValOptions
from data import create_dataset
from models import create_model
import torch
import pandas as pd
from collections import OrderedDict

if __name__ == '__main__':
    # 解析命令行参数或配置文件中的选项，返回一个包含训练配置的对象 opt。这些选项可能包括数据集路径、模型类型、训练超参数等。
    optTrain = TrainOptions().parse()   # get training options
    # optVal = ValOptions().parse()   # get training options

    dataset_train = create_dataset(optTrain)  # create a dataset given opt.dataset_mode and other options
    dataset_train_size = len(dataset_train)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_train_size)

    model = create_model(optTrain)      # create a model given opt.model and other options
    model.setup(optTrain)               # regular setup: load and print networks; create schedulers 加载预训练参数
    total_iters = 0                # the total number of training iterations

    # dataset_val = create_dataset(optVal)
    # dataset_val_size = len(dataset_val)

    # 创建一个DataFrame来存储损失信息
    loss_df1 = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss_Name', 'Loss_Value'])
    # loss_df2 = pd.DataFrame(columns=['Epoch', 'Iteration', 'Loss_Name', 'Loss_Value'])

    for epoch in range(optTrain.epoch_count, optTrain.n_epochs + optTrain.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        if epoch % optTrain.save_epoch_freq == 0 or epoch == optTrain.n_epochs + optTrain.n_epochs_decay: 
            # 初始化一个空的累加字典
            lossesTrain = OrderedDict()
            for name in model.loss_names:
                lossesTrain[name] = 0.0  # 初始化每个损失项的累加值为 0

        for i, data in enumerate(dataset_train):  # inner loop within one epoch 处理一个epoch内的每个batch
            total_iters += optTrain.batch_size
            epoch_iter += optTrain.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing 解压缩数据并预处理
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights 计算梯度、优化参数 更新网络权重

            if epoch % optTrain.save_epoch_freq == 0 or epoch == optTrain.n_epochs + optTrain.n_epochs_decay: 
                current_losses = model.get_current_losses()
                # 累加每个损失项
                for name, value in current_losses.items():
                    lossesTrain[name] += value
            
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        # cache our model every <save_epoch_freq> epochs
        if epoch % optTrain.save_epoch_freq == 0 or epoch == optTrain.n_epochs + optTrain.n_epochs_decay: 
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest') #存储网络模型
            model.save_networks(epoch)
            for name, loss_value in lossesTrain.items():
                loss_value =loss_value/dataset_train_size
                print(f"{name}: {loss_value}")
                new_row = {'Epoch': epoch, 'Iteration': total_iters, 'Loss_Name': name, 'Loss_Value': loss_value}
                loss_df1 = pd.concat([loss_df1, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        print('End of epoch %d / %d \t' % (epoch, optTrain.n_epochs + optTrain.n_epochs_decay))
        
        # # TODO 修改频次 在每个epoch结束后添加验证步骤
        # if epoch % optVal.validation_freq == 0 or epoch == optTrain.n_epochs + optTrain.n_epochs_decay:
        #     model.eval()  # 设置模型为评估模式
        #     optTrain.isTrain=False
        #     lossesVal = OrderedDict()
        #     for name in model.loss_names:
        #         lossesVal[name] = 0.0  # 初始化每个损失项的累加值为 0
        #     with torch.no_grad():  # 不需要计算梯度
        #         for val_data in dataset_val:
        #             model.set_input(val_data)
        #             model.forward()  # 前向传播
        #             model.calculate_loss_D()
        #             model.calculate_loss_G()
        #             current_losses = model.get_current_losses()
        #             # 累加每个损失项
        #             for name, value in current_losses.items():
        #                 lossesVal[name] += value
        
        #     for name, loss_value in lossesVal.items():
        #         loss_value =loss_value/dataset_val_size
        #         print(f"{name}: {loss_value}")
        #         new_row = {'Epoch': epoch, 'Iteration': total_iters, 'Loss_Name': name, 'Loss_Value': loss_value}
        #         loss_df2 = pd.concat([loss_df2, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        
        #     optTrain.isTrain=True  # 重新设置模型为训练模式
        #     print('Val End of epoch %d / %d \t' % (epoch, optTrain.n_epochs + optTrain.n_epochs_decay))

    # 在训练循环结束后写入训练损失
    training_loss_file = 'training_losses.xlsx'
    loss_df1.to_excel(training_loss_file, index=False)
    print(f"Training losses saved to {training_loss_file}")

    # # 在训练循环结束后写入验证损失
    # validation_loss_file = 'validation_losses.xlsx'
    # loss_df2.to_excel(validation_loss_file, index=False)
    # print(f"Validation losses saved to {validation_loss_file}")
