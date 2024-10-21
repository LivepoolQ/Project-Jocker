"""
@Author: Ziqian Zou
@Date: 2024-10-19 15:59:02
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-21 19:59:28
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
import qpid

from .__args import GroupModelArgs
from .gp import GroupModel, GroupStructure

# Register new args and models
qpid.register_args(GroupModelArgs, 'GroupModel Args')
qpid.register(
    gp=[GroupStructure, GroupModel],
)
