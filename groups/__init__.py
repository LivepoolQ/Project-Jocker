"""
@Author: Ziqian Zou
@Date: 2024-10-19 15:59:02
@LastEditors: Ziqian Zou
@LastEditTime: 2024-11-22 16:22:21
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""
import qpid
from qpid.mods.vis import __vis
from qpid.utils import get_relative_path

from .__args import GroupModelArgs
from .gp import GroupModel, GroupStructure

# Register new args and models
qpid.register_args(GroupModelArgs, 'GroupModel Args')
qpid.register(
    gp=[GroupStructure, GroupModel],
    gp_msn=[GroupStructure, GroupModel]
)


__vis.OBS_IMAGE = get_relative_path(__file__, './static/obs_small_gp.png')
__vis.NEI_OBS_IMAGE = get_relative_path(__file__, './static/neighbor_small_gp.png')
__vis.CURRENT_IMAGE = get_relative_path(__file__, './static/neighbor_current_gp.png')
__vis.GT_IMAGE = get_relative_path(__file__, './static/gt_small_gp.png')
__vis.PRED_IMAGE = get_relative_path(__file__, './static/pred_small_gp.png')
