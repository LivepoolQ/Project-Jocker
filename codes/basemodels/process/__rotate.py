"""
@Author: Conghao Wong
@Date: 2022-09-01 11:15:52
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-30 16:57:43
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...constant import INPUT_TYPES, OUTPUT_TYPES
from ...utils import ROTATE_BIAS, batch_matmul
from .__base import BaseProcessLayer


class Rotate(BaseProcessLayer):
    """
    Rotate trajectories to the reference angle.
    The default reference angle is 0.
    """

    def __init__(self, anntype: str, ref,
                 *args, **kwargs):

        super().__init__(anntype, ref,
                         preprocess_input_types=[INPUT_TYPES.OBSERVED_TRAJ],
                         postprocess_input_types=[OUTPUT_TYPES.PREDICTED_TRAJ],
                         *args, **kwargs)

        if self.picker.base_dim != 2:
            raise NotImplementedError(f'Rotate is not supported on {anntype}.')

    def update_paras(self, inputs: dict[str, tf.Tensor]) -> None:
        trajs = inputs[INPUT_TYPES.OBSERVED_TRAJ]
        steps = trajs.shape[-2]
        vectors = (tf.gather(trajs, steps-1, axis=-2) -
                   tf.gather(trajs, 0, axis=-2))

        angles = []
        for [x, y] in self.order:
            vector_x = tf.gather(vectors, x, axis=-1)
            vector_y = tf.gather(vectors, y, axis=-1)
            main_angle = tf.atan((vector_y + ROTATE_BIAS) /
                                 (vector_x + ROTATE_BIAS))
            angle = self.ref - main_angle
            angles.append(angle)

        self.paras = angles

    def preprocess(self, inputs: dict[str, tf.Tensor],
                   use_new_paras=True) -> dict[str, tf.Tensor]:
        """
        Rotate trajectories to the reference angle.
        """
        if use_new_paras:
            self.update_paras(inputs)

        angles = self.paras
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.rotate(_input, angles, inverse=False)
        return outputs

    def postprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Rotate trajectories back to their original angles.
        """
        angles = self.paras
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.rotate(_input, angles, inverse=True)
        return outputs

    def rotate(self, trajs: tf.Tensor,
               ref_angles: list[tf.Tensor],
               inverse=False):

        ndim = trajs.ndim

        trajs_rotated = []
        for angle, [x, y] in zip(ref_angles, self.order):
            if inverse:
                angle = -1.0 * angle

            rotate_matrix = tf.stack([[tf.cos(angle), tf.sin(angle)],
                                      [-tf.sin(angle), tf.cos(angle)]])

            if ndim >= 3:
                # transpose to (batch, 2, 2)
                rotate_matrix = tf.transpose(
                    rotate_matrix, list(tf.range(2, rotate_matrix.ndim)) + [0, 1])

            while rotate_matrix.ndim < ndim:
                rotate_matrix = tf.expand_dims(rotate_matrix, -3)

            _trajs = tf.gather(trajs, [x, y], axis=-1)
            _trajs_rotated = batch_matmul(_trajs, rotate_matrix)
            trajs_rotated.append(_trajs_rotated)

        trajs_rotated = tf.concat(trajs_rotated, axis=-1)
        return trajs_rotated
