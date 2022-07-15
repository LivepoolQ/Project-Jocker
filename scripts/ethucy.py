"""
@Author: Conghao Wong
@Date: 2022-07-15 14:45:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-15 15:37:35
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Any
import numpy as np
import os
import plistlib

SCALE = 720.0

SOURCE_FILE = './data/ethucy/{}/true_pos_.csv'
TARGET_FILE = './data/ethucy/{}/ann.csv'
BASE_DIR = './datasets'
SUBSETS_DIR = './datasets/subsets'

SUBSETS: dict[str, Any] = {}

SUBSETS['eth'] = dict(
    name='eth',
    dataset_dir='./data/eth/univ',
    order=[1, 0],
    paras=[6, 25],
    video_path='./videos/eth.mp4',
    weights=[[
        [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
        [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
        [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
    ], 0.65, 225, 0.6, 160],
    scale=1,
)

SUBSETS['hotel'] = dict(
    name='hotel',
    dataset_dir='./data/eth/hotel',
    order=[0, 1],
    paras=[10, 25],
    video_path='./videos/hotel.mp4',
    weights=[[
        [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
        [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
        [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
    ], 0.54, 470, 0.54, 300],
    scale=1,
)

SUBSETS['zara1'] = dict(
    name='zara1',
    dataset_dir='./data/ucy/zara/zara01',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara1.mp4',
    weights=[-42.54748107, 580.5664891, 47.29369894, 3.196071003],
    scale=1,
)

SUBSETS['zara2'] = dict(
    name='zara2',
    dataset_dir='./data/ucy/zara/zara02',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara2.mp4',
    weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
    scale=1,
)

SUBSETS['univ'] = dict(
    name='univ',
    dataset_dir='./data/ucy/univ/students001',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[-41.1428, 576, 48, 0],
    scale=1,
)

SUBSETS['zara3'] = dict(
    name='zara3',
    dataset_dir='./data/ucy/zara/zara03',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara2.mp4',
    weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
    scale=1,
)

SUBSETS['univ3'] = dict(
    name='univ3',
    dataset_dir='./data/ucy/univ/students003',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[-41.1428, 576, 48, 0],
    scale=1,
)

SUBSETS['unive'] = dict(
    name='unive',
    dataset_dir='./data/ucy/univ/uni_examples',
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[-41.1428, 576, 48, 0],
    scale=1,
)

TESTSETS = ['eth', 'hotel', 'zara1', 'zara2', 'univ']


def write_plist(value: dict, path: str):
    with open(path, 'wb+') as f:
        plistlib.dump(value, f)


def transform_annotations():
    """"
    Transform annotations with the new `ann.csv` type.
    """
    for name in SUBSETS.keys():

        source = SOURCE_FILE.format(name)
        target = TARGET_FILE.format(name)

        data_original = np.loadtxt(source, delimiter=',')
        r = data_original[2:].T

        weights = SUBSETS[name]['weights']
        order = SUBSETS[name]['order']

        if len(weights) == 4:
            result = np.column_stack([
                weights[2] * r.T[1] + weights[3],
                weights[0] * r.T[0] + weights[1],
            ])/SCALE

        else:
            H = weights[0]
            real = np.ones([r.shape[0], 3])
            real[:, :2] = r
            pixel = np.matmul(real, np.linalg.inv(H))
            pixel = pixel[:, :2]
            result = np.column_stack([
                weights[1] * pixel.T[0] + weights[2],
                weights[3] * pixel.T[1] + weights[4],
            ])/SCALE

        dat = np.column_stack([data_original[0].astype(int).astype(str),
                               data_original[1].astype(int).astype(str),
                               result.T[order[0]].astype(str),
                               result.T[order[1]].astype(str)])

        with open(target, 'w+') as f:
            for _dat in dat:
                f.writelines([','.join(_dat)+'\n'])
        print('{} Done.'.format(target))


def save_dataset_info():
    """
    Save dataset information into `plist` files.
    """
    subsets = {}
    for name, value in SUBSETS.items():
        subsets[name] = dict(
            name=name,
            dataset_dir='./data/ethucy/{}'.format(name),
            paras=value['paras'],
            video_path=value['video_path'],
            weights=SCALE,
            scale=1,
            dimension=2,
            anntype='coordinate',
        )

    for path in [BASE_DIR, SUBSETS_DIR]:
        if not os.path.exists(path):
            os.mkdir(path)

    for ds in TESTSETS:
        train_sets = []
        test_sets = []
        val_sets = []

        for d in subsets.keys():
            if d == ds:
                test_sets.append(d)
                val_sets.append(d)
            else:
                train_sets.append(d)

        write_plist({'train': train_sets,
                     'test': test_sets,
                     'val': val_sets,
                     'weights': SCALE,
                     'dimension': 2,
                     'anntype': 'coordinate'},
                    os.path.join(BASE_DIR, '{}.plist'.format(ds)))

    for key, value in subsets.items():
        write_plist(value,
                    p := os.path.join(SUBSETS_DIR, '{}.plist'.format(key)))
        print('Successfully saved at {}'.format(p))


if __name__ == '__main__':
    transform_annotations()
    save_dataset_info()