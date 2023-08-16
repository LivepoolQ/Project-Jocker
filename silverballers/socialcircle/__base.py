"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-16 14:59:34
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

from codes.managers import Structure

from ..agents import BaseAgentModel, BaseAgentStructure
from .__args import SocialCircleArgs


class BaseSocialCircleModel(BaseAgentModel):
    def __init__(self, Args: SocialCircleArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.args: SocialCircleArgs

    def print_info(self, **kwargs):
        info = {'Transform type (SocialCircle)': self.args.Ts,
                'Partitions in SocialCircle': self.args.partitions,
                'Max partitions in SocialCircle': self.args.obs_frames}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseSocialCircleStructure(BaseAgentStructure):
    ARG_TYPE = SocialCircleArgs

    def __init__(self, terminal_args: Union[list[str], SocialCircleArgs],
                 manager: Structure = None):
        super().__init__(terminal_args, manager)

        if self.args.model_type != 'agent-based':
            self.log('SocialCircle models only support model type `agent-based`.' +
                     f' Current setting is `{self.args.model_type}`.',
                     level='error', raiseError=ValueError)
