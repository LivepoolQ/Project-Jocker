import torch

import qpid
from qpid.constant import INPUT_TYPES
from qpid.model import layers
from qpid.model.layers import LinearLayerND
from qpid.utils import INIT_POSITION

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
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
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

        # Concat fc layer
        self.concat_fc = layers.Dense(
            self.gp_args.output_units * 4, self.gp_args.output_units * 4, activation=nn.Tanh)

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

        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(
            self.gp_args.output_units * 8, self.gp_args.generation_num, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(
            self.gp_args.output_units * 4, self.gp_args.output_units * 8)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(
            self.gp_args.output_units * 8, self.gp_args.output_units * 8, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.gp_args.output_units * 8,
                                        self.args.pred_frames * self.dim)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)
        # if self.gp_args.no_social and not training:
        #     nei = self.create_empty_neighbors(obs)
        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        if self.gp_args.use_group:
            # Long term distance between neighbors and obs(ade)
            long_term_dis = c_nei - c_obs[:, None, ...]
            # final step distance(fde)
            final_vec = c_nei[..., -1:, :] - c_obs[:, None, -1:, :]
            group_mask = ((torch.sum(long_term_dis ** 2,
                                     dim=[-1, -2]) < 6).to(dtype=torch.int32)) * ((torch.sum(final_vec ** 2, dim=[-1, -2]) < 6/8).to(dtype=torch.int32))
            trajs_group = (
                nei * group_mask[..., None, None]).to(dtype=torch.float32)
            group_num = torch.sum(group_mask, dim=-1)

        # group trajectory encoding
        if self.gp_args.use_group:

            # Obs trajectory encoding
            f_obs = self.te(obs)
            f_group = self.te(trajs_group)
            f_group = (
                torch.sum(f_group * group_mask[..., None, None], dim=1) + 1e-8) / \
                (group_num[..., None, None] + 1e-8)

            # Concat obs and nei feature
            f = torch.concat([f_obs, f_group], dim=-1)

        else:
            f_obs = self.te2(obs)
            f = f_obs

        # Compute Conception and padding
        conception_circle = self.cl.implement(self, inputs)
        f_social = self.tse(conception_circle)
        f_social = torch.repeat_interleave(f_social, torch.tensor(
            f_obs.shape[-2]).to(f_obs.device).to(torch.int32), dim=-2)
        # f_social = torch.zeros_like(f_social)
        # Concat feature of sc and traj
        _f = torch.concat([f_social, f], dim=-1)

        f = self.concat_fc(_f)

        # # -----------------------
        # # The following lines are used to draw visualized figures in our paper
        # from scripts.draw_fov import draw_contributions

        # con = torch.sum(f_social).numpy()
        # obs_f = torch.sum(f_obs).numpy()
        # group_f = torch.sum(f_group).numpy()

        # w = self.concat_fc.linear.weight
        # d = self.gp_args.output_units

        # _con_f = _f[..., :d*2]
        # _obs_f = _f[..., d*2:d*3]
        # _group_f = _f[..., d*3:d*4]
        # w_con = w[..., :d*2]
        # w_obs = w[..., d*2:d*3]
        # w_group = w[..., d*3:d*4]

        # con_f=_con_f @ w_con.T
        # obs_f=_obs_f @ w_obs.T
        # group_f=_group_f @ w_group.T

        # draw_contributions(con_f=_con_f @ w_con.T,
        #                    obs_f=_obs_f @ w_obs.T,
        #                    group_f=_group_f @ w_group.T,
        #                    file_name='contributions',
        #                    color_high=[0xff, 0xa6, 0x00],
        #                    color_low=[0x00, 0x3f, 0x5c],
        #                    max_width=0.3, min_width=0.2)
        # Vis codes end here
        # -----------------------

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        obs_lin = self.lp(obs)
        obs_lin = torch.concat([obs, obs_lin], dim=-2)

        # (batch, obs+pred, out_uni * 4)
        f_tran, _ = self.bb(inputs=f, targets=obs_lin, training=training)
        # (batch, pred, out_uni * 4)
        f_tran = f_tran[:, self.args.obs_frames:, ...]

        # Prediction
        for _ in range(repeats):
            # Assign random ids and embedding
            z = torch.normal(mean=0, std=1, size=list(
                f_tran.shape[:-1]) + [self.d_id])
            # (batch, pred, out_uni * 4)
            f_z = self.ie(z.to(obs.device))

            # (batch, pred, out_uni * 8)
            f_final = torch.concat([f_tran, f_z], dim=-1)

            # Multiple generations -> (batch, Kc, out_uni * 8)
            # (batch, steps, Kc)
            adj = self.ms_fc(f_final)
            adj = torch.transpose(adj, -1, -2)
            # (batch, Kc, out_uni * 4)
            f_multi = self.ms_conv(f_tran, adj)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y, list(y.shape[:-1]) +
                              [self.args.pred_frames, self.dim])

            all_predictions.append(y)

        Y = torch.concat(all_predictions, dim=-3)
        return Y

    def create_empty_neighbors(self, ego_traj: torch.Tensor):
        empty = INIT_POSITION * torch.ones([ego_traj.shape[0],
                                            self.args.max_agents - 1,
                                            ego_traj.shape[-2],
                                            ego_traj.shape[-1]]).to(ego_traj.device)
        return torch.concat([ego_traj[..., None, :, :], empty], dim=-3)


class GroupStructure(qpid.training.Structure):
    MODEL_TYPE = GroupModel
