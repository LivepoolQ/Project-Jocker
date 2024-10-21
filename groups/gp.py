"""
@Author: Ziqian Zou
@Date: 2024-10-18 16:58:13
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-21 19:59:46
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""

import torch

import qpid
from qpid.constant import INPUT_TYPES
from socialCircle import SocialCircleArgs
from socialCircle.__layers import SocialCircleLayer

from .__args import GroupModelArgs
from .__traj_encoding import TrajEncoding

nn = torch.nn


class GroupModel(qpid.model.Model):
    """
    """

    def __init__(self, structure=None, *args, **kwargs):

        super().__init__(structure, *args, **kwargs)

        # Init args
        self.gp_args = self.args.register_subargs(GroupModelArgs, 'gp_args')
        self.sc_args = self.args.register_subargs(SocialCircleArgs, 'sc')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Trajectory encoding
        self.te = TrajEncoding(output_units=self.gp_args.output_units,
                               input_units=self.dim)

        # social_circle encoding
        self.tse = TrajEncoding(
            output_units=self.gp_args.output_units * 2, input_units=3)

        # SocialCircle layer
        self.sc = SocialCircleLayer(
            partitions=self.args.obs_frames,
            max_partitions=self.args.obs_frames,
        )

        # Noise encoding
        self.ie = TrajEncoding(self.d, self.d_id)

        # Backbone
        self.bb = qpid.model.transformer.Transformer(
            num_layers=4,
            d_model=self.args.feature_dim,
            num_heads=8,
            dff=512,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.obs_frames,
            include_top=False
        )

        # Final layer
        self.fl = torch.nn.Sequential(
            torch.nn.Linear(self.args.feature_dim *
                            self.args.obs_frames, self.args.feature_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.feature_dim * 2,
                            self.args.pred_frames * 2),
            torch.nn.Tanh(),
        )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        # Long term distance between neighbors and obs
        long_term_dis = c_nei - c_obs[:, None, ...]
        group_mask = (torch.sum(long_term_dis ** 2,
                      dim=[-1, -2]) < 10).to(dtype=torch.int32)
        trajs_group = c_nei * group_mask[..., None, None]
        group_num = torch.sum(group_mask, dim=-1)

        # Compute SocialCircle
        social_circle = self.sc.implement(self, inputs)
        f_social = self.tse(social_circle)

        # Obs trajectory encoding
        f_obs = self.te(obs)

        # group trajectory encoding
        f_group = self.te(trajs_group)
        f_group = (torch.sum(f_group, dim=1) + 1e-5) / \
            (group_num[:, None, None] + 1e-5)

        # Concat obs and nei feature
        f = torch.concat([f_obs, f_group], dim=-1)

        # Concat feature of sc and traj
        f = torch.concat([f_social, f], dim=-1)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        f_tran, _ = self.bb(inputs=f, targets=obs, training=training)

        # Prediction
        for _ in range(repeats):
            # Assign random ids and embedding
            z = torch.normal(mean=0, std=1, size=list(
                f.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            f_final = f_tran + f_z

            g = torch.flatten(f_final, start_dim=1, end_dim=-1)
            g = self.fl(g)
            traj_pred = g[:, None, ...].reshape(
                f.shape[0], 1, self.args.pred_frames, -1)

            all_predictions.append(traj_pred)

        Y = torch.concat(all_predictions, dim=-3)
        return Y


class GroupStructure(qpid.training.Structure):
    MODEL_TYPE = GroupModel
