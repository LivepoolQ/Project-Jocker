"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-15 17:19:38
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as qpid

from . import original_models
from .__args import PhysicalCircleArgs, SocialCircleArgs
from .ev_sc import EVSCModel, EVSCStructure

# Add new args
qpid.register_args(SocialCircleArgs, 'SocialCircle Args')
qpid.register_args(PhysicalCircleArgs, 'PhysicalCircle Args')
qpid.add_arg_alias(alias=['--sc', '-sc', '--socialCircle'],
                   command=['--model', 'MKII', '--loads'],
                   pattern='{},speed')

# Register Circle-based models
qpid.register(
    # SocialCircle Models
    evsc=[EVSCStructure, EVSCModel],
)
