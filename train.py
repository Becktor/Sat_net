import torch
import references.detection.utils as utils
from references.detection.engine import train_one_epoch, evaluate
from dataset import SatDataset, PennFudanDataset, get_transform
from models import get_model_instance_segmentation
import wandb
import PIL
import numpy as np


def main():
    wandb.init(project="Satellite", config={
        "learning_rate": 1e-4,
        "gamma": 0.1,
        "split": 0.2,
        "batch_size": 8,
    })
    config = wandb.config
    wandb_name = wandb.run.name + "_" + wandb.run.id

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = SatDataset('Datasets/seg', get_transform(train=True))
    dataset_test = SatDataset('Datasets/seg', get_transform(train=False))

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
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=50,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_log = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        val_log, img, out = evaluate(model, data_loader_test, device=device)
        img = wandb.Image(img[0])
        masks = wandb.Image(out[0]['masks'])
        wandb.log({"val/img": img,
                   "val/masks": masks,
                   "train/lr": train_log.meters['lr'].value,
                   "train/clf_loss": train_log.meters['loss_classifier'].value,
                   "train/bbox_loss": train_log.meters['loss_box_reg'].value,
                   "train/mask_loss": train_log.meters['loss_mask'].value,
                   "train/loss_obj": train_log.meters['loss_objectness'].value,
                   "train/loss_rpn_box_reg": train_log.meters['loss_rpn_box_reg'].value})

    print("That's it!")


if __name__ == "__main__":
    main()
