"""
@Author: Ziqian Zou
@Date: 2024-10-18 17:03:36
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-19 16:29:16
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class GroupModelArgs(EmptyArgs):

    @property
    def use_group(self) -> int:
        """
        Choose whether to use pedestrian groups when calculating SocialCircle.
        """
        return self._arg('use_group', 1, argtype=STATIC, desc_in_model_summary='use_group_model')

    @property
    def output_units(self) -> int:
        """
        Set number of the output units of trajectory encoding.
        """
        return self._arg('output_units', 32, argtype=STATIC)
