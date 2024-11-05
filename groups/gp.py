"""
@Author: Ziqian Zou
@Date: 2024-10-18 16:58:13
@LastEditors: Ziqian Zou
@LastEditTime: 2024-11-04 20:53:48
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2024 Ziqian Zou, All Rights Reserved.
"""

import torch

import qpid
from qpid.constant import INPUT_TYPES
from qpid.model.layers import LinearLayerND

from .__args import GroupModelArgs
from .__traj_encoding import TrajEncoding
from .conception import ConceptionLayer

nn = torch.nn


class GroupModel(qpid.model.Model):
    """
    """

    def __init__(self, structure=None, *args, **kwargs):

        super().__init__(structure, *args, **kwargs)

        # Init args
        self.gp_args = self.args.register_subargs(GroupModelArgs, 'gp_args')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Trajectory encoding
        self.te = TrajEncoding(output_units=self.gp_args.output_units,
                               input_units=self.dim)
        self.te2 = TrajEncoding(output_units=self.gp_args.output_units * 2,
                                input_units=self.dim)

        # social_circle encoding
        self.tse = TrajEncoding(output_units=self.gp_args.output_units * 2,
                                input_units=7)

        # Conception layer
        self.cl = ConceptionLayer(use_view_angle=self.gp_args.use_view_angle,
                                  view_angle=self.gp_args.view_angle,
                                  use_pooling=self.gp_args.use_pooling,
                                  use_max=self.gp_args.use_max,
                                  use_velocity=self.gp_args.use_velocity,
                                  use_distance=self.gp_args.use_distance,
                                  use_move_dir=self.gp_args.use_move_dir,
                                  use_group=self.gp_args.use_group,
                                  disable_conception=self.gp_args.disable_conception)

        # Noise encoding
        self.ie = TrajEncoding(self.d, self.d_id)

        # Obs encoded as target of transformer
        self.pe = TrajEncoding(self.args.pred_frames *
                               self.dim, self.args.obs_frames * self.dim)

        # Linear prediction of obs as the target of transformer
        self.lp = LinearLayerND(
            self.args.obs_frames, self.args.pred_frames, return_full_trajectory=False)

        # Backbone
        self.bb = qpid.model.transformer.Transformer(
            num_layers=4,
            d_model=self.args.feature_dim,
            num_heads=8,
            dff=512,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.pred_frames + self.args.obs_frames,
            include_top=False
        )

        # Final layer
        self.fl = torch.nn.Sequential(
            torch.nn.Linear(self.args.feature_dim * 2, self.args.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.feature_dim,
                            self.dim),
            torch.nn.Tanh(),
        )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        if self.gp_args.use_group:
            # Long term distance between neighbors and obs(ade)
            long_term_dis = c_nei - c_obs[:, None, ...]
            # final step distance(fde)
            final_vec = c_nei[..., -1:, :] - c_obs[:, None, -1:, :]
            group_mask = ((torch.sum(long_term_dis ** 2,
                                    dim=[-1, -2]) < 6).to(dtype=torch.int32)) * ((torch.sum(final_vec ** 2, dim=[-1, -2]) < 6/self.args.obs_frames).to(dtype=torch.int32))
            trajs_group = (
                nei * group_mask[..., None, None]).to(dtype=torch.float32)
            group_num = torch.sum(group_mask, dim=-1)


        # group trajectory encoding
        if self.gp_args.use_group:

            # Obs trajectory encoding
            f_obs = self.te(obs)
            f_group = self.te(trajs_group)
            f_group = (torch.sum(f_group, dim=1) + 1) / \
                (group_num[..., None, None] + 1)

            # Concat obs and nei feature
            f = torch.concat([f_obs, f_group], dim=-1)

        else:
            f_obs = self.te2(obs)
            f = f_obs

        # Compute Conception and padding
        conception_circle = self.cl.implement(self, inputs)
        f_social = self.tse(conception_circle)
        f_social = torch.repeat_interleave(f_social, torch.tensor(f_obs.shape[-2]).to(f_obs.device).to(torch.int32), dim=-2)

        # Concat feature of sc and traj
        f = torch.concat([f_social, f], dim=-1)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        obs_lin = self.lp(obs)
        obs_lin = torch.concat([obs, obs_lin], dim=-2)

        f_tran, _ = self.bb(inputs=f, targets=obs_lin, training=training)
        f_tran = f_tran[:, self.args.obs_frames:, ...]

        # Prediction
        for _ in range(repeats):
            # Assign random ids and embedding
            z = torch.normal(mean=0, std=1, size=list(
                f_tran.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            f_final = torch.concat([f_tran, f_z], dim=-1)

            g = self.fl(f_final)
            traj_pred = g[:, None, ...]

            all_predictions.append(traj_pred)

        Y = torch.concat(all_predictions, dim=-3)
        return Y


class GroupStructure(qpid.training.Structure):
    MODEL_TYPE = GroupModel
