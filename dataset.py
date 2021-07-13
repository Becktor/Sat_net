import os
import numpy as np
import torch
import torchvision
from PIL import Image
import csv

from typing import Tuple

import references.detection.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from future.utils import raise_from
import sys
import matplotlib.pyplot as plt
from utils import *
import re


class CSVDataset(object):
    """CSV dataset."""

    def __init__(self, root, train_file, use_path=True, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = os.path.join(root, train_file)
        self.transform = transform
        self.root = root
        # csv with img_path, roll_sun, pitch_sun, yaw_sun, x, y, z, roll_tar, pitch_tar, yaw_tar
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data, self.image_target = self._read_data(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    @staticmethod
    def _parse(value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    @staticmethod
    def _open_for_csv(path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx, debug=False):
        img = torch.as_tensor(self.load_image(idx), dtype=torch.float32).permute(2, 0, 1)
        name = self.image_names[idx]
        angles = torch.as_tensor(list(self.image_data[name].values()))
        target = torch.as_tensor(list(self.image_target[name].values()))
        rot = torch.deg2rad(target[-3:])
        # img = center_crop(img, name)
        try:
            raise Exception("not working")
        except Exception as e:
            print(e)

        if debug:
            plot_image_and_target(img, rot, show=True)
        if self.transform:
            img, target = self.transform(img, target)

        return img, angles, rot

    def load_image(self, image_index):
        img_path = os.path.join(self.root, self.image_names[image_index])
        img = np.array(Image.open(img_path).convert("RGB"))

        return img.astype(np.float32) / 255.0

    def _read_data(self, csv_reader):
        result = {}
        sun_angles = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                img_file, rs, ps, ys, x, y, z, rt, pt, yt = row[:10]

                # If a row contains only an image path, it's an image without annotations.
                if (rs, ps, ys, x, y, z, rt, pt, yt) == ('', '', '', '', '', '', '', '', ''):
                    continue

                rs = self._parse(rs, float, 'line {}: malformed rs: {{}}'.format(line))
                ps = self._parse(ps, float, 'line {}: malformed ps: {{}}'.format(line))
                ys = self._parse(ys, float, 'line {}: malformed ys: {{}}'.format(line))
                x = self._parse(x, float, 'line {}: malformed x: {{}}'.format(line))
                y = self._parse(y, float, 'line {}: malformed y: {{}}'.format(line))
                z = self._parse(z, float, 'line {}: malformed z: {{}}'.format(line))
                rt = self._parse(rt, float, 'line {}: malformed rt: {{}}'.format(line))
                pt = self._parse(pt, float, 'line {}: malformed pt: {{}}'.format(line))
                yt = self._parse(yt, float, 'line {}: malformed yt: {{}}'.format(line))

                result[img_file] = {'x': x, 'y': y, 'z': z, 'roll': rt, 'pitch': pt, 'yaw': yt}
                sun_angles[img_file] = {'roll': rs, 'pitch': ps, 'yaw': ys}

            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file, rs, ps, ys, x, y, z, rt, pt, yt\' or \'img_file,,,,,,,\''.format(
                        line)),
                    None)
        return sun_angles, result

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class KeypointDataset(object):
    """CSV dataset."""

    def __init__(self, root, train_file, use_path=True, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.target_type = "Gaussian"
        self.train_file = os.path.join(root, train_file)
        self.transform = transform
        self.root = root
        self.souyz_width = 10
        self.fov = 50
        self.input_size = 400
        # csv with img_path, roll_sun, pitch_sun, yaw_sun, x, y, z, roll_tar, pitch_tar, yaw_tar
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data, self.image_target = self._read_data(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    @staticmethod
    def _parse(value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    @staticmethod
    def _open_for_csv(path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx, debug=False):
        img = torch.as_tensor(self.load_image(idx), dtype=torch.float32).permute(2, 0, 1)
        name = self.image_names[idx]
        sun_angles = torch.as_tensor(list(self.image_data[name].values()))
        kp_target = torch.as_tensor(list(self.image_target[name].values()))
        img, kp_target = self.center_crop(img, name, kp_target)
        img, kp_target = self.as_percent(img, kp_target)
        if self.transform:
            img, target = self.transform(img, kp_target)

        return img, sun_angles, kp_target

    def load_image(self, image_index) -> np.array:
        img_path = os.path.join(self.root, self.image_names[image_index])
        img = np.array(Image.open(img_path).convert("RGB"))

        return img.astype(np.float32) / 255.0

    def as_percent(self, img, kp_target):
        kp_target[:, :2] = self.input_size / kp_target[:, :2]
        return img, kp_target[:, :2]

    def _read_data(self, csv_reader):
        result = {}
        sun_angles = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                img_file, r, p, y, a1, a2, a3, a4, a5, = row[:9]

                # If a row contains only an image path, it's an image without annotations.
                if (a1, a2, a3, a4, a5, r, p, y) == ('', '', '', '', '', '', '', ''):
                    continue

                a1 = np.fromstring(a1[1:-1], sep=" ")
                a2 = np.fromstring(a2[1:-1], sep=" ")
                a3 = np.fromstring(a3[1:-1], sep=" ")
                a4 = np.fromstring(a4[1:-1], sep=" ")
                a5 = np.fromstring(a5[1:-1], sep=" ")

                r = self._parse(r, float, 'line {}: malformed r: {{}}'.format(line))
                p = self._parse(p, float, 'line {}: malformed p: {{}}'.format(line))
                y = self._parse(y, float, 'line {}: malformed y: {{}}'.format(line))

                result[img_file] = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, }
                sun_angles[img_file] = {'roll': r, 'pitch': p, 'yaw': y}

            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'img_file,r ,p , y, a1, a2, a3, a4, a5 \' or \'img_file,,,,,,,\''.format(
                        line)),
                    None)
        return sun_angles, result

    def generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', 'Only supports gaussian map for now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)

    def sat_size(self, dist, img_size, fov, size):
        factor = (img_size[2] / 2) / (np.tan(np.deg2rad(fov / 2)) * dist)
        size = size * factor
        return size

    def sat_size_and_target_reshape(self, img, dist, kp_target):
        img_shape = img.shape
        size = int(self.sat_size(dist, img_shape, self.fov, self.souyz_width) * 1.1)
        x_red = int((img_shape[2] - size) / 2)
        y_red = int((img_shape[1] - size) / 2)
        for x in kp_target:
            x[0] -= x_red
            x[1] -= y_red
        return size, kp_target

    def center_crop(self, img, name, kp_target):
        dist = int(re.sub('\D', "", name[-9:-4])) / 100
        size, kp_target = self.sat_size_and_target_reshape(img, dist, kp_target)
        img = F.center_crop(img, [size, size])
        img = F.resize(img, [self.input_size, self.input_size], InterpolationMode.BILINEAR)
        kp_target *= self.input_size / size
        return img, kp_target


class SatDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)
        mask[mask > 0] = 1

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
