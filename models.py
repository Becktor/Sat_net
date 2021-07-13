import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
import numpy as np


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class ConvBlock(nn.Module):
    def __init__(self, in_size=None, feature_size=32):
        super(ConvBlock, self).__init__()
        if in_size is None:
            in_size = feature_size
        self.block = nn.Sequential(
            nn.Conv2d(in_size, feature_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(feature_size)
        )

    def forward(self, x):
        return self.block(x)


class EulerLoss(torch.nn.Module):

    def __init__(self):
        super(EulerLoss, self).__init__()
        self.pi = torch.tensor(np.pi)

    def forward(self, pred, tar):
        loss = torch.abs(pred - tar)
        loss[loss > self.pi] = 2 * self.pi - loss[loss > self.pi]
        return torch.mean(loss)


class CosineSimilarity(torch.nn.Module):

    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.pi = torch.tensor(np.pi)
        self.eps = 1e-6
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=self.eps)

    def forward(self, pred, tar):
        loss = 1 - self.cos(pred, tar)
        return torch.mean(loss)


class DotLoss(torch.nn.Module):

    def __init__(self):
        super(DotLoss, self).__init__()
        self.pi = torch.tensor(np.pi)
        self.eps = 1e-6
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=self.eps)

    def forward(self, pred, tar):
        loss = (pred * tar).sum(1) / torch.linalg.norm(tar, dim=1) ** 2
        minus_loss = 1 - loss
        abs_loss = torch.abs(minus_loss)
        return torch.mean(abs_loss)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        assert not target.requires_grad
        assert pred.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class SimpleNetwork(nn.Module):
    def __init__(self, input_size):
        super(SimpleNetwork, self).__init__()
        # here we add the individual layers
        self.input_size = input_size
        self.angles_to_image = nn.Linear(3, input_size * input_size)  # INPUT SIZE is 400 FIX HARDCODED
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.linear = nn.Linear((256 * 7 * 7), 5 * 2 * 3)  # 5x2 points to predict x 3 quantiles
        self.pi = torch.tensor(np.pi)
        layers = [self.conv1,
                  ConvBlock(16, 32), self.max_pool,
                  ConvBlock(32, 32), self.max_pool,
                  ConvBlock(32, 32), self.max_pool,
                  ConvBlock(32, 64), self.max_pool,
                  ConvBlock(64, 64), self.max_pool,
                  ConvBlock(64, 128), self.max_pool,
                  ConvBlock(128, 256)]
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(*layers)
        self.quantiles = (0.1, 0.5, 0.9)
        self.loss = QuantileLoss(quantiles=self.quantiles)  # nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def forward(self, x, ang, y) -> torch.tensor:
        ang = self.angles_to_image(ang).reshape(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, ang], 1)
        output = self.net(x)
        # print(output.shape)
        flat = self.flatten(output)
        output = self.linear(flat)
        output = torch.clamp(output, 0, self.input_size)
        output = torch.reshape(output, [-1, 3, 5 * 2])
        y_flat = self.flatten(y).float()
        if self.training:
            return self.loss(output, y_flat), output
        else:
            return output
