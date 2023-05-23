"""
@Author: Conghao Wong
@Date: 2022-06-23 10:23:53
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-23 16:25:18
@Description: Second stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers
from codes.basemodels.transformer import Transformer

from ..__args import HandlerArgs
from .__baseHandler import BaseHandlerModel, BaseHandlerStructure


class VBModel(BaseHandlerModel):
    """
    Spectrum Interpolation Sub-network
    ---

    The second stage V^2-Net sub-network.
    It is used to interpolate agents' entire predictions
    by considering their interactions details.
    It also implements on agents' spectrums instead of
    their trajectories.
    """

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler, key_points,
                         structure, *args, **kwargs)

        # Transform layers
        input_steps = self.args.obs_frames
        output_steps = self.args.obs_frames + self.args.pred_frames

        Tlayer, ITlayer = layers.get_transform_layers(self.args.T)
        self.t_layer = Tlayer((input_steps, self.dim))
        self.it_layer = ITlayer((output_steps, 2))

        # Shapes
        input_Tsteps, Tchannels = self.t_layer.Tshape
        output_Tsteps, _ = self.it_layer.Tshape

        # Linear layer
        self.linear_int = layers.interpolation.LinearPositionInterpolation()

        # Encoding layers
        # NOTE: All the following layers are calculated
        #       in the ***frequency domain***.
        self.te = layers.TrajEncoding(units=self.d//2,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.t_layer)

        self.ce = layers.ContextEncoding(units=self.d//2,
                                         output_channels=input_Tsteps,
                                         activation=tf.nn.tanh)

        self.transformer = Transformer(num_layers=4,
                                       d_model=self.d,
                                       num_heads=8,
                                       dff=512,
                                       input_vocab_size=None,
                                       target_vocab_size=Tchannels,
                                       pe_input=input_Tsteps,
                                       pe_target=output_Tsteps,
                                       include_top=True)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        # unpack inputs
        trajs_md, maps = inputs[:2]
        keypoints_md = keypoints

        # Only accept 2-dimensional trajectories
        if training:
            trajs = trajs_md
            keypoints = keypoints_md
        else:
            picker = self.structure.get_manager(BaseHandlerStructure).picker
            trajs = picker.get_center(trajs_md)[..., :2]
            keypoints = picker.get_center(keypoints)[..., :2]

        # Embedding and encoding
        # Transformations are applied in `self.te`
        traj_feature = self.te.call(trajs)    # (batch, input_Tsteps, d//2)
        context_feature = self.ce.call(maps)  # (batch, input_Tsteps, d//2)

        # transformer inputs shape = (batch, input_Tsteps, d)
        t_inputs = tf.concat([traj_feature, context_feature], axis=-1)

        # transformer target shape = (batch, output_Tsteps, Tchannels)
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # Add the last obs point to finish linear interpolation
        linear_pred = self.linear_int.call(keypoints_index, keypoints)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        t_outputs = self.t_layer.call(traj)

        # transformer output shape = (batch, output_Tsteps, Tchannels)
        p_fft, _ = self.transformer.call(t_inputs,
                                         t_outputs,
                                         training=training)

        # Inverse transform
        p = self.it_layer.call(p_fft)
        y = p[:, self.args.obs_frames:, :]

        if training:
            return y

        # Calculate linear prediction (M-dimensional)
        keypoints_md = tf.concat([trajs_md[:, -1:, :], keypoints_md], axis=1)
        l = self.linear_int.call(keypoints_index, keypoints_md)

        # Linear center points
        l_center = picker.get_center(l)[tf.newaxis]
        l_co = tf.cast(picker.get_coordinate_series(l), tf.float32)

        # Bias to the center points
        bias_center = l_co - l_center
        bias_linear = (y - linear_pred)[tf.newaxis]
        new_center = y[tf.newaxis]

        y_md = new_center + bias_center + bias_linear   # (M, batch, pred, 2)
        y_md = tf.concat(list(y_md), axis=-1)
        return y_md


class VB(BaseHandlerStructure):
    """
    Training structure for the second stage sub-network
    """

    def __init__(self, terminal_args: list[str],
                 manager=None,
                 is_temporary=False):

        super().__init__(terminal_args, manager, is_temporary)
        self.set_model_type(VBModel)
