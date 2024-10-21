"""
@Author: Ziqian Zou
@Date: 2024-10-18 16:24:29
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-21 19:59:35
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""

import torch

nn = torch.nn


class TrajEncoding(nn.Module):
    """

    """

    def __init__(self,
                 output_units: int,
                 input_units: int = 2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_units = output_units
        self.input_units = input_units

        self.fc1 = nn.Linear(input_units, output_units)
        self.ac1 = nn.ReLU()

        self.fc2 = nn.Linear(output_units, output_units)
        self.ac2 = nn.Tanh()

    def forward(self, trajs: torch.Tensor) -> torch.Tensor:

        f = self.fc1(trajs)
        f = self.ac1(f)
        f = self.fc2(f)
        trajs_enc = self.ac2(f)

        return trajs_enc
