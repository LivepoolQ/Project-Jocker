"""
@Author: Conghao Wong
@Date: 2023-09-06 20:45:28
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 18:02:34
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .__agent47C import Agent47C, Agent47CModel
from .__MiniV import MinimalV, MinimalVModel
from .__MSNalpha import MSNAlpha, MSNAlphaModel
from .__Valpha import VA, VAModel

__qpid.silverballers.register(
    # MSN
    msna=[MSNAlpha, MSNAlphaModel],

    # V^2-Net
    va=[VA, VAModel],
    agent=[VA, VAModel],

    # E-V^2-Net
    eva=[Agent47C, Agent47CModel],

    # agent47 series
    agent47C=[Agent47C, Agent47CModel],

    # Other models
    mv=[MinimalV, MinimalVModel],
)
