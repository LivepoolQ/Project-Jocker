"""
@Author: Conghao Wong
@Date: 2022-06-20 20:10:58
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-14 16:36:42
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
import os
import plistlib
import time

import numpy as np
import tensorflow as tf

"""
Configs
"""
# Basic parameters
TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

# Paths settings
ROOT_TEMP_DIR = './temp_files'
DATASET_CONFIG_DIR = './dataset_configs'
ANNOTATION_CONFIGS_FILE = './codes/annSettings.plist'

# Dataset configs
INIT_POSITION = 100000000

# Context map configs
SEG_IMG = 'segmentation_image'
RGB_IMG = 'rgb_image'

# WINDOW_EXPAND_PIXEL = 0.3
# WINDOW_SIZE_PIXEL = 200.0
WINDOW_EXPAND_PIXEL = 10.0
WINDOW_SIZE_PIXEL = 10.0

WINDOW_EXPAND_METER = 10.0
WINDOW_SIZE_METER = 10.0

MAP_HALF_SIZE = 50  # Local map's half size
AVOID_SIZE = 15     # Avoid size in grid cells when modeling social interaction
INTEREST_SIZE = 20  # Interest size in grid cells when modeling social interaction

POOLING_BEFORE_SAVING = True

# Preprocess configs
ROTATE_BIAS = 0.01
SCALE_THRESHOLD = 0.05

# Visualization configs
SMALL_POINTS = True
OBS_IMAGE = './figures/obs_small.png' if SMALL_POINTS else './figures/obs.png'
NEIGHBOR_IMAGE = './figures/neighbor_small.png' if SMALL_POINTS else './figures/neighbor.png'
GT_IMAGE = './figures/gt_small.png' if SMALL_POINTS else './figures/gt.png'
PRED_IMAGE = './figures/pred_small.png' if SMALL_POINTS else './figures/pred.png'
DISTRIBUTION_IMAGE = './figures/dis.png'

# Log paths and configs
LOG_FILE = './test.log'
LOG_MAX_LIST_LEN = 10
LOG_STREAM_HANDLER = logging.StreamHandler()

# Weights configs
WEIGHTS_FORMAT = '.tf'
CHECKPOINT_FILENAME = 'best_ade_epoch.txt'

# Visualization settings
# color bar in BGR format
# rgb(0, 0, 178) -> rgb(252, 0, 0) -> rgb(255, 255, 10)
DISTRIBUTION_COLORBAR = np.column_stack([
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([178, 0, 10])),
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([0, 0, 255])),
    np.interp(np.arange(256),
              np.array([0, 127, 255]),
              np.array([0, 252, 255])),
])

# Whether draw lines and text in images and videos
DRAW_LINES = False
DRAW_TEXT_IN_VIDEOS = False
DRAW_TEXT_IN_IMAGES = True


def dir_check(target_dir: str) -> str:
    """
    Used for checking if the `target_dir` exists.
    If it does not exist, it will make it.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    return target_dir


def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.

    :param path: The path of the plist file.
    :return dat: A `dict` object loaded from the file.
    """
    with open(path, 'rb') as f:
        dat = plistlib.load(f)

    return dat


def get_mask(input: tf.Tensor, dtype=tf.float32):
    return tf.cast(input < 0.05 * INIT_POSITION, dtype)


def get_loss_mask(obs: tf.Tensor, label: tf.Tensor, return_numpy=False):
    """
    Get mask from both model predictions and labels.
    Return type: `tf.float32`.

    :param obs: Observed trajectories, shape = `(..., steps, dim)`
    :param label: Label trajectories, shape = `(..., steps, dim)`
    """
    pred_mask = get_mask(tf.reduce_sum(obs, axis=[-1, -2]))
    label_mask = get_mask(tf.reduce_sum(label, axis=[-1, -2]))
    mask = pred_mask * label_mask
    if return_numpy:
        mask = mask.numpy()
    return mask


def batch_matmul(a: tf.Tensor, b: tf.Tensor, *args, **kwargs):
    """
    Run matmul operations on a batch of inputs.
    Other args will be wrapped to `tf.matmul`.

    :param a: Input, shape is `(..., a, b)`.
    :param b: Another input, shape is `(..., b, c)`.
    """
    if a.ndim <= 4:
        return tf.matmul(a, b, *args, **kwargs)

    batch = tf.shape(a)[:-3]
    _a = tf.reshape(a, [-1]+list(tf.shape(a)[2:]))
    _b = tf.reshape(b, [-1]+list(tf.shape(b)[2:]))
    res = tf.matmul(_a, _b, *args, **kwargs)

    return tf.reshape(res, list(batch) + list(tf.shape(res)[1:]))
