"""
@Author: Ziqian Zou
@Date: 2024-10-18 17:03:36
@LastEditors: Ziqian Zou
@LastEditTime: 2024-11-05 21:30:13
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
import numpy as np

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class GroupModelArgs(EmptyArgs):

    @property
    def use_group(self) -> int:
        """
        Choose whether to use pedestrian groups when calculating SocialCircle.
        """
        return self._arg('use_group', 1, argtype=STATIC, desc_in_model_summary='use_group_model')

    @property
    def view_angle(self) -> float:
        """
        Value of conception view field.
        """
        return self._arg('view_angle', np.pi, argtype=STATIC)

    @property
    def use_view_angle(self) -> int:
        """
        Choose whether to use view angle in calculating conception.
        """
        return self._arg('use_view_angle', 1, argtype=STATIC)

    @property
    def use_pooling(self) -> int:
        """
        Choose whether to use pooling in calculating conception value.
        Only choose one between pooling and max.
        """
        return self._arg('use_pooling', 1, argtype=STATIC)

    @property
    def use_max(self) -> int:
        """
        Choose whether to use max in calculating conception value.
        Only choose one between pooling and max.
        """
        return self._arg('use_max', 0, argtype=STATIC)

    @property
    def output_units(self) -> int:
        """
        Set number of the output units of trajectory encoding.
        """
        return self._arg('output_units', 32, argtype=STATIC)

    @property
    def use_velocity(self) -> int:
        """
        Choose whether to use the velocity factor in the conception.
        """
        return self._arg('use_velocity', 1, argtype=STATIC)

    @property
    def use_distance(self) -> int:
        """
        Choose whether to use the distance factor in the conception.
        """
        return self._arg('use_distance', 1, argtype=STATIC)

    @property
    def use_move_dir(self) -> int:
        """
        Choose whether to use the move direction factor in the conception.
        """
        return self._arg('use_move_dir', 1, argtype=STATIC)

    @property
    def disable_conception(self) -> int:
        """
        Choose whether to disable conception layer in the GroupModel.
        """
        return self._arg('disable_conception', 0, argtype=STATIC)

    @property
    def generation_num(self) -> int:
        """
        Number of multi-style generation.
        """
        return self._arg('generation_num', 20, argtype=STATIC)
    
    @property
    def no_social(self) -> int:
        return self._arg('no_social', 0, argtype=TEMPORARY)
