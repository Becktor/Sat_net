import os
from __future__ import annotations
import math
import sys
import torch
import references.detection.utils as utils
from references.detection.engine import train_one_epoch, evaluate
from dataset import SatDataset, PennFudanDataset, get_transform, CSVDataset, KeypointDataset
from models.simple_models import get_model_instance_segmentation, SimpleNetwork, AutoEncoderNetwork
import wandb
import PIL
import numpy as np
from utils import plot_image_and_target, plot_image_target_pred, count_parameters, channel_to_color, save_ckp, load_ckp
from collections import deque
from datetime import datetime
from distinctipy import distinctipy
import matplotlib.pyplot as plt


def main():
    wandb.init(project="Satellite_DOF", config={
        "learning_rate": 1e-5,
        "gamma": 0.1,
        "split": 0.2,
        "batch_size": 8,
    })
    config = wandb.config
    wandb_name = wandb.run.name + "_" + wandb.run.id
    checkpoint_dir = os.path.join('trained_models', wandb_name)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = KeypointDataset('Datasets/KP_48k_30_50_70', 'JoinedData.csv', is_gaussian=True)

    dataset_test = KeypointDataset('Datasets/KP_48k_30_50_70', 'JoinedData.csv', is_gaussian=True)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    test_split = int(len(indices) * config.split)
    dataset = torch.utils.data.Subset(dataset, indices[:-test_split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_split:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = AutoEncoderNetwork(input_size=400)
    count_parameters(model)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                               step_size=20,
    #                                               gamma=0.1)
    n_iters = len(data_loader)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-5,
    #                                                 step_size_up=n_iters, cycle_momentum=False)
    wr = 10
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, n_iters * wr, 2)

    # let's train it for 620 epochs
    num_epochs = 620
    img = 0
    lbl = 0
    output = 0
    curr_lr = 0
    queue = deque()
    epoch_queue = deque()
    plot = False
    number_of_keypoints = 5
    colors = distinctipy.get_colors(number_of_keypoints)
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        # train for one epoch, printing every 10 iterations
        model.train()
        mean_loss = []
        mean_val_loss = []
        strlen = 0
        # if epoch == (wr-1):
        # lr_scheduler.base_lrs[0] = lr_scheduler.base_lrs[0] * 0.5
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        for iter_num, data in enumerate(data_loader):
            iter_start_time = datetime.now()
            images, angles, labels = data

            images = torch.stack(images).to(device)
            angles = torch.stack(angles).to(device)
            labels = torch.stack(labels).to(device)
            losses, pred = model(images, angles, labels)
            mean_loss.append(float(losses))
            if len(queue) > 0:
                d_time = np.mean(queue)
                str = 'E: {} i: {}/{} - Est rem ept: {:1.2f} -- ' \
                      'curr loss: {:1.5f}, mean loss: {:1.5f}'.format(epoch, iter_num, n_iters,
                                                                      d_time * (len(data_loader) - iter_num),
                                                                      float(losses), np.mean(mean_loss))
                print(str, end='\r')
                strlen = len(str)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            delta_time = datetime.now() - iter_start_time
            queue.appendleft(delta_time.total_seconds())
            if len(queue) > 100:
                queue.pop()

        print(" " * strlen, end='\r')
        print("Epoch {} loss: {:1.5f}".format(epoch, np.mean(mean_loss)))
        # validation
        test_len = len(data_loader_test)
        v_img, v_lbl, v_out = 0, 0, 0

        for iter_num, data in enumerate(data_loader_test):
            images, angles, labels = data
            images = torch.stack(images).to(device)
            angles = torch.stack(angles).to(device)
            labels = torch.stack(labels).to(device)
            val_losses, pred = model(images, angles, labels)
            mean_val_loss.append(float(val_losses))
            print('val step: {}/{}'.format(iter_num, test_len), end='\r')
            if iter_num == 0:
                v_img, v_lbl, v_out = images, labels, pred

        print('Validation Mean loss {:1.3f}'.format(np.mean(mean_val_loss)))

        # # Writing to wandb log
        log_dict = {"train/loss": np.mean(mean_loss),
                    "val/loss": np.mean(mean_val_loss),
                    "optim/lr": curr_lr}

        pred_in_numpy = v_out[0].detach().cpu().numpy()
        images, joined = channel_to_color(pred_in_numpy, colors)
        img_in_numpy = v_img[0].permute(1, 2, 0).detach().cpu().numpy()
        log_dict["pred"] = wandb.Image(joined + img_in_numpy)
        for x, i in enumerate(images):
            log_dict["pred_kp_{}".format(x)] = wandb.Image(i)
        # normalize
        norm_pred = (pred_in_numpy - np.min(pred_in_numpy)) / (np.max(pred_in_numpy) - np.min(pred_in_numpy))
        n_images, n_joined = channel_to_color(norm_pred, colors)
        log_dict["n_pred"] = wandb.Image(joined + img_in_numpy)
        for x, i in enumerate(n_images):
            log_dict["n_pred_kp_{}".format(x)] = wandb.Image(i)

        if epoch == 0:
            lbl_in_numpy = v_lbl[0].detach().cpu().numpy()
            images, joined = channel_to_color(lbl_in_numpy, colors)
            log_dict["lbl"] = wandb.Image(joined)
            for x, i in enumerate(images):
                log_dict["lbl_kp_{}".format(x)] = wandb.Image(i)

        wandb.log(log_dict)
        delta_time = datetime.now() - epoch_start_time
        epoch_queue.appendleft(delta_time.total_seconds())
        d_time = np.mean(epoch_queue)
        print("epoch runtime: {:1.3f}. --- est time remaining {:1.3f}".format(d_time, d_time * (num_epochs - epoch)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(mean_loss)
        }
        save_ckp(checkpoint, model, checkpoint_dir)
    print("That's it!")


if __name__ == "__main__":
    main()
