"""
@Author: Ziqian Zou
@Date: 2024-11-07 11:27:32
@LastEditors: Ziqian Zou
@LastEditTime: 2024-11-11 22:10:16
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from qpid.utils import dir_check

ROOT_DIR = './temp_files/gp'

COLOR_HIGH = [250, 100, 100]
COLOR_LOW = [60, 100, 220]

MAX_WIDTH = 0.3
MIN_WIDTH = 0.1


def cal_color(weights: list[float], color_high: list[int], color_low: list[int]):
    w: np.ndarray = np.array(weights)
    w = w - w.min()
    w /= (w.max() - w.min())
    _color_high = np.array(color_high)
    _color_low = np.array(color_low)
    color = _color_low + w[:, np.newaxis] * (_color_high - _color_low)
    return [(i.tolist()) for i in color/255]


def cal_radius(weights: list[float],
               max_value=MAX_WIDTH,
               min_value=MIN_WIDTH) -> np.ndarray:
    w: np.ndarray = np.array(weights)
    return (max_value - min_value) * w/w.max() + min_value

def cal_radius_group(weights: list[float], max_value=MAX_WIDTH) -> np.ndarray:
    w: np.ndarray = np.array(weights)
    return w/np.sum(w)

def draw_fov(con: torch.Tensor, file_name: str,
                    color_high: list[int] = COLOR_HIGH,
                    color_low: list[int] = COLOR_LOW,
                    max_width: float = MAX_WIDTH,
                    min_width: float = MIN_WIDTH,
                    view_angle: float = 180,
                    start_angle: float = 0):
    p = con.squeeze().numpy()
    view_angle = view_angle
    fig = plt.figure()
    colors = cal_color(p, color_high, color_low)
    radius = cal_radius(p, max_width, min_width)
    

    plt.pie(x=[view_angle/2, view_angle/2, 360-view_angle],
            radius=1, counterclock=True, colors=colors, startangle=start_angle)
    
    for index, r in enumerate(radius):
            _colors: list = [(0, 0, 0, 0) for _ in p]
            _colors[index] = fig.get_facecolor()
            plt.pie(x=[view_angle/2, view_angle/2, 360-view_angle], 
                    counterclock=True, startangle=start_angle,
                    radius=1.0-r,
                    colors=_colors)
    
    r = dir_check(ROOT_DIR)
    plt.savefig(f := (os.path.join(r, f'{file_name}.png')))
    plt.close()

    # Save as a png image
    fig_saved: np.ndarray = cv2.imread(f)
    alpha_channel = 255 * (np.min(fig_saved[..., :3], axis=-1) != 255)
    fig_png = np.concatenate(
        [fig_saved, alpha_channel[..., np.newaxis]], axis=-1)

    # Cut the image
    areas = fig_png[..., -1] == 255
    x_value = np.sum(areas, axis=0)
    x_index_all = np.where(x_value)[0]
    y_value = np.sum(areas, axis=1)
    y_index_all = np.where(y_value)[0]

    cv2.imwrite(os.path.join(r, f'{file_name}_cut.png'),
                fig_png[y_index_all[0]:y_index_all[-1],
                        x_index_all[0]:x_index_all[-1]])
    
def draw_contributions(con_f: torch.Tensor, obs_f: torch.Tensor, 
                       group_f: torch.Tensor, file_name: str,
                    color_high: list[int] = COLOR_HIGH,
                    color_low: list[int] = COLOR_LOW,
                    max_width: float = MAX_WIDTH,
                    min_width: float = MIN_WIDTH):
     
    con = torch.sum(con_f[0] ** 2, dim=[-1, -2]).numpy()
    obs = torch.sum(obs_f[0] ** 2, dim=[-1, -2]).numpy()
    group = torch.sum(group_f[0] ** 2, dim=[-1, -2]).numpy()
    p = np.array([con/con_f.shape[-2], group, obs]).tolist()
    fig = plt.figure()
    colors = cal_color(p, color_high, color_low)
    radius = cal_radius_group(p, max_width)
    radius_sum = sum(radius)
    colors_white = [1.0, 1.0, 1.0]

    
    plt.pie(x=[radius[0], radius_sum - radius[0]], colors=["#58508D", "#FFFFFF"], radius=1, startangle=180, counterclock=False)
    plt.pie(x=[radius[1], radius_sum - radius[1]], colors=["#DE5A79", "#FFFFFF"], radius=1 - 0.2, startangle=180, counterclock=False)
    plt.pie(x=[radius[2], radius_sum - radius[2]], colors=["#FFA600", "#FFFFFF"], radius=1 - 0.4, startangle=180, counterclock=False)
    plt.pie(x=[1], colors=["#FFFFFF"], radius=1 - 0.6)

    # plt.pie(x=[1, 3], colors=["#58508D", "#FFFFFF"], radius=1, startangle=180, counterclock=False)
    # plt.pie(x=[1, 3], colors=["#DE5A79", "#FFFFFF"], radius=1 - 0.2, startangle=180, counterclock=False)
    # plt.pie(x=[1, 3], colors=["#FFA600", "#FFFFFF"], radius=1 - 0.4, startangle=180, counterclock=False)
    # plt.pie(x=[1], colors=["#FFFFFF"], radius=1 - 0.6)

    r = dir_check(ROOT_DIR)
    plt.savefig(f := (os.path.join(r, f'{file_name}.png')))
    plt.close()

    # Save as a png image
    fig_saved: np.ndarray = cv2.imread(f)
    alpha_channel = 255 * (np.min(fig_saved[..., :3], axis=-1) != 255)
    fig_png = np.concatenate(
        [fig_saved, alpha_channel[..., np.newaxis]], axis=-1)

    # Cut the image
    areas = fig_png[..., -1] == 255
    x_value = np.sum(areas, axis=0)
    x_index_all = np.where(x_value)[0]
    y_value = np.sum(areas, axis=1)
    y_index_all = np.where(y_value)[0]

    cv2.imwrite(os.path.join(r, f'{file_name}_cut.png'),
                fig_png[y_index_all[0]:y_index_all[-1],
                        x_index_all[0]:x_index_all[-1]])