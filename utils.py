from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import cos, sin, sqrt, acos, radians
from PIL import Image
import os
from distinctipy import distinctipy

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_image_and_target(img, rpy, show=False, tmpname='tmp.png', inrad=True):
    img = img.permute(1, 2, 0)
    plt.imshow(img.numpy())
    x, y, z = rot_vecs(rpy, inrad)
    arr_width = 3
    scale = 100
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, x[1] * scale, x[0] * scale, fc='pink', ec='pink',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, y[1] * scale, y[0] * scale, fc='lightgreen', ec='lightgreen',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, z[1] * scale, z[0] * scale, fc='lightblue', ec='lightblue',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.title("rpy: {}".format(rpy))
    if show:
        plt.show()
        return 0

    arr = plt.savefig(tmpname)
    plt.clf()
    img = Image.open(tmpname).convert("RGB")
    return img


def plot_image_target_pred(img, rpy, prpy, show=False, tmp_name='tmp.png', in_rad=True):
    img = img.permute(1, 2, 0)
    plt.imshow(img.numpy())
    x, y, z = rot_vecs(rpy, in_rad)
    xp, yp, zp = rot_vecs(prpy, in_rad)
    arr_width = 3
    scale = 100
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, x[1] * scale, x[0] * scale, fc='red', ec='red',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, xp[1] * scale, xp[0] * scale, fc='pink', ec='pink',
              head_width=3 * arr_width, head_length=6 * arr_width, alpha=0.5)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, y[1] * scale, y[0] * scale, fc='green', ec='green',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, yp[1] * scale, yp[0] * scale, fc='lightgreen',
              ec='lightgreen', head_width=3 * arr_width, head_length=6 * arr_width, alpha=0.5)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, z[1] * scale, z[0] * scale, fc='blue', ec='blue',
              head_width=3 * arr_width, head_length=6 * arr_width)
    plt.arrow(img.shape[0] // 2, img.shape[1] // 2, zp[1] * scale, zp[0] * scale, fc='lightblue', ec='lightblue',
              head_width=3 * arr_width, head_length=6 * arr_width, alpha=0.5)
    plt.title("tar: {} \n pred: {} \n diff: {}".format(np.rad2deg(rpy.numpy()), np.rad2deg(prpy), np.abs(rpy.numpy() - prpy)))
    if show:
        plt.show()
        return 0

    plt.savefig(tmp_name)
    plt.clf()
    img = Image.open(tmp_name).convert("RGB")
    return img


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def q_conjugate(q):
    w, x, y, z = q
    return w, -x, -y, -z


def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z


def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta


def rot_vecs(rpy, inrad=True):
    roll, pitch, yaw = rpy
    if not inrad:
        roll = radians(roll)
        pitch = radians(pitch)
        yaw = radians(yaw)
    x_axis_unit = (1, 0, 0)
    y_axis_unit = (0, 1, 0)
    z_axis_unit = (0, 0, 1)
    r1 = axisangle_to_q(x_axis_unit, roll)
    r2 = axisangle_to_q(y_axis_unit, pitch)
    r3 = axisangle_to_q(z_axis_unit, yaw)

    # xxx
    y = qv_mult(r1, y_axis_unit)
    y = qv_mult(r2, y)
    y = qv_mult(r3, y)
    # yyy
    x = qv_mult(r1, x_axis_unit)
    x = qv_mult(r2, x)
    x = qv_mult(r3, x)
    # zzz
    z = qv_mult(r1, z_axis_unit)
    z = qv_mult(r2, z)
    z = qv_mult(r3, z)
    z = -1 * np.array(z)  # to be consistent with view in unreal

    return x, y, z


def channel_to_color(grey_imgs, colors):
    images = []
    joined = np.zeros([grey_imgs.shape[1], grey_imgs.shape[2], 3])

    for x, g in enumerate(grey_imgs):
        reshaped = g.reshape(g.shape[0], g.shape[1], 1)
        conc = np.concatenate([reshaped, reshaped, reshaped], axis=2)
        colored = conc * colors[x]
        images.append(colored)
        joined += colored
    joined[joined > 1] = 1
    return images, joined


def save_ckp(state, model, is_best, checkpoint_dir, epoch):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model_{}.pt'.format(epoch))
        torch.save(state, best_filepath)


def load_ckp(checkpoint_filepath, model, optimizer=None, scheduler=None):
    cwd = os.path.join(os.getcwd(), checkpoint_filepath)
    path = os.path.join(cwd, 'checkpoint.pt')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint