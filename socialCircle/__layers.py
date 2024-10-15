"""
@Author: Conghao Wong
@Date: 2023-08-08 14:55:56
@LastEditors: Ziqian Zou
@LastEditTime: 2024-10-15 17:57:50
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, process
from qpid.utils import get_mask

from .__args import PhysicalCircleArgs, SocialCircleArgs

NORMALIZED_SIZE = None

INF = 1000000000
SAFE_THRESHOLDS = 0.05
MU = 0.00000001


class SocialCircleLayer(torch.nn.Module):
    """
    A layer to compute SocialCircle meta components.

    Supported factors:
    - Velocity;
    - Distance;
    - Direction;
    - Movement Direction (Optional).
    """

    def __init__(self, partitions: int,
                 max_partitions: int,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_direction: bool | int = True,
                 use_move_direction: bool | int = False,
                 use_acceleration: bool | int = False,
                 mu=0.0001,
                 relative_velocity: bool | int = False,
                 output_units=128,
                 *args, **kwargs):
        """
        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param max_partitions: The number of partitions (after zero padding).

        ## SocialCircle Meta Components
        :param use_velocity: Choose whether to use the `velocity` factor.
        :param use_distance: Choose whether to use the `distance` factor.
        :param use_direction: Choose whether to use the `direction` factor.
        :param use_move_direction: Choose whether to use the `move direction` factor.
        :param use_acceleration: Choose whether to use the `acceleration` factor

        ## SocialCircle Options
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing. \
            It only works when `relative_velocity` is set to `True`.
        :param output_units: The hidden and output layer's units of 4 factors' encoders.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.rel_velocity = relative_velocity
        self.use_move_direction = use_move_direction
        self.use_acceleration = use_acceleration
        self.mu = mu
        output_units = output_units // 2
        # encoder of velocity and acceleration
        self.vel_acc_fc1 = layers.Dense(
            int(self.use_velocity) + int(self.use_acceleration), output_units, torch.nn.ReLU)
        self.vel_acc_fc2 = layers.Dense(
            output_units, output_units, torch.nn.Tanh)
        # encoder of distance and direction
        self.dis_dir_fc1 = layers.Dense(
            int(self.use_distance) + int(self.use_direction), output_units, torch.nn.ReLU)
        self.dis_dir_fc2 = layers.Dense(
            output_units, output_units, torch.nn.Tanh)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        nei_vector_last = nei_trajs[..., -1, :] - nei_trajs[..., -2, :]
        nei_vector_second_last = nei_trajs[..., -2, :] - nei_trajs[..., -3, :]
        obs_vector_last = trajs[..., -1:, :] - trajs[..., -2:-1, :]
        obs_vector_second_last = trajs[..., -2:-1, :] - trajs[..., -3:-2, :]

        # Velocity factor
        if self.use_velocity:
            # Calculate velocities
            nei_velocity = torch.norm(nei_vector, dim=-1)    # (batch, n)
            obs_velocity = torch.norm(obs_vector, dim=-1)    # (batch, 1)

            # Speed factor in the SocialCircle
            if self.rel_velocity:
                f_velocity = (nei_velocity + self.mu)/(obs_velocity + self.mu)
            else:
                f_velocity = nei_velocity

        # Distance factor
        if self.use_distance:
            f_distance = torch.norm(nei_posion_vector, dim=-1)

        # Move direction factor
        if self.use_move_direction:
            obs_move_direction = torch.atan2(obs_vector[..., 0],
                                             obs_vector[..., 1])
            nei_move_direction = torch.atan2(nei_vector[..., 0],
                                             nei_vector[..., 1])
            delta_move_direction = nei_move_direction - obs_move_direction
            f_move_direction = delta_move_direction % (2*np.pi)

        # Direction factor
        f_direction = torch.atan2(nei_posion_vector[..., 0],
                                  nei_posion_vector[..., 1])
        f_direction = f_direction % (2*np.pi)

        # acceleration factor
        if self.use_acceleration:
            # calculate the velocity change from the 2nd last step to the last step
            nei_acc_vector = nei_vector_last - nei_vector_second_last
            obs_acc_vector = obs_vector_last - obs_vector_second_last
            nei_acc = torch.norm(nei_acc_vector, dim=-1)    # (batch, n)
            obs_acc = torch.norm(obs_acc_vector, dim=-1)    # (batch, 1)
            f_acceleration = nei_acc

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_trajs, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        sc_vel_acc = []
        sc_dis_dir = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001
            sc_vel_acc.append([])
            sc_dis_dir.append([])

            if self.use_velocity:
                _velocity = torch.sum(f_velocity * _mask, dim=-1) / n
                sc_vel_acc[-1].append(_velocity)

            if self.use_distance:
                _distance = torch.sum(f_distance * _mask, dim=-1) / n
                sc_dis_dir[-1].append(_distance)

            if self.use_direction:
                _direction = torch.sum(f_direction * _mask, dim=-1) / n
                sc_dis_dir[-1].append(_direction)

            if self.use_move_direction:
                _move_d = torch.sum(f_move_direction * _mask, dim=-1) / n
                sc_vel_acc[-1].append(_move_d)

            if self.use_acceleration:
                _acceleration = torch.sum(f_acceleration * _mask, dim=-1) / n
                sc_vel_acc[-1].append(_acceleration)

        # shape of vel_acc factor (batch, partitions, 2)
        sc_vel_acc = [torch.stack(i, dim=-1) for i in sc_vel_acc]
        sc_vel_acc = torch.stack(sc_vel_acc, dim=-2)

        # shape of dis_dir factor (batch, partitions, 2)
        sc_dis_dir = [torch.stack(i, dim=-1) for i in sc_dis_dir]
        sc_dis_dir = torch.stack(sc_dis_dir, dim=-2)

        # encode vel_acc and dis_dir factor
        f_vel_acc = self.vel_acc_fc2(self.vel_acc_fc1(sc_vel_acc))
        f_dis_dir = self.dis_dir_fc2(self.dis_dir_fc1(sc_dis_dir))

        fusion_feature = torch.concat([f_vel_acc, f_dis_dir], dim=-1)

        fusion_feature = self.pad(fusion_feature)
        return fusion_feature

    def implement(self, model: Model, inputs: list[torch.Tensor]):
        """
        Compute the SocialCircle from original model inputs.
        """
        # Unpack inputs
        # (batch, obs, dim)
        obs = model.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, a:=max_agents, obs, dim)
        nei = model.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Start computing the SocialCircle
        # SocialCircle will be computed on each agent's center point
        c_obs = model.picker.get_center(obs)[..., :2]
        c_nei = model.picker.get_center(nei)[..., :2]

        # Compute and encode the SocialCircle
        social_circle = self(c_obs, c_nei)

        # Set all partitions to zeros (counterfactual variations)
        sc_args = model.args.register_subargs(SocialCircleArgs, 'sc')
        if sc_args.use_empty_neighbors:
            return torch.zeros_like(social_circle)

        return social_circle

    def pad(self, input: torch.Tensor):
        """
        Zero-padding the input tensor (whose shape must be `(batch, steps, dim)`).
        It will pad the input tensor on the `steps` axis if `steps < max_partitions`,
        where the `max_partitions` is usually the maximum one of either the number of
        observation steps or the number of SocialCircle partitions.
        """
        current_steps = input.shape[-2]
        target_steps = max(self.max_partitions, self.partitions)
        if ((p := target_steps - current_steps) > 0):
            paddings = [0, 0, 0, p, 0, 0]
            return torch.nn.functional.pad(input, paddings)
        else:
            return input
