"""
@Author: Conghao Wong
@Date: 2022-06-20 16:14:03
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-07 11:25:25
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import re

import numpy as np
import tensorflow as tf

from ..args import Args
from ..utils import CHECKPOINT_FILENAME, WEIGHTS_FORMAT
from . import process

MOVE = 'MOVE'
ROTATE = 'ROTATE'
SCALE = 'SCALE'
UPSAMPLING = 'UPSAMPLING'


class Model(tf.keras.Model):
    """
    Model
    -----

    Usage
    -----
    When training or test new models, please subclass this class, and clarify
    model layers used in your model.
    ```python
    class MyModel(Model):
        def __init__(self, Args, structure, *args, **kwargs):
            super().__init__(Args, structure, *args, **kwargs)

            self.fc = tf.keras.layers.Dense(64, tf.nn.relu)
            self.fc1 = tf.keras.layers.Dense(2)
    ```

    Then define your model's pipeline in `call` method:
    ```python
        def call(self, inputs, training=None, mask=None):
            y = self.fc(inputs)
            return self.fc1(y)
    ```

    Public Methods
    --------------
    ```python
    # forward model with pre-process and post-process
    (method) forward: (self: Model, model_inputs: list[Tensor], training=None, *args, **kwargs) -> list[Tensor]

    # Pre/Post-processes
    (method) pre_process: (self: Model, tensors: list[Tensor], training=None, use_new_para_dict=True, *args, **kwargs) -> list[Tensor]
    (method) post_process: (self: Model, outputs: list[Tensor], training=None, *args, **kwargs) -> list[Tensor]
    ```
    """

    def __init__(self, Args: Args,
                 structure=None,
                 *args, **kwargs):

        super().__init__()
        self.args = Args
        self.structure = structure

        # preprocess
        self.processor: process.BaseProcessLayer = None
        self._default_process_para = {MOVE: Args.pmove,
                                      SCALE: Args.pscale,
                                      ROTATE: Args.protate}

    def call(self, inputs,
             training=None,
             *args, **kwargs):

        raise NotImplementedError

    def forward(self, inputs: list[tf.Tensor],
                training=None) -> list[tf.Tensor]:
        """
        Run a forward implementation.

        :param inputs: input tensor (or a `list` of tensors)
        :param training: config if running as training or test mode
        :return outputs_p: model's output. type=`list[tf.Tensor]`
        """

        inputs_p = self.process(inputs, preprocess=True, training=training)
        outputs = self(inputs_p, training=training)
        outputs_p = self.process(outputs, preprocess=False, training=training)
        return outputs_p

    def set_preprocess(self, **kwargs):
        """
        Set pre-process methods used before training.

        args: pre-process methods.
            - Move positions on the observation step to (0, 0):
                args in `['Move', ...]`

            - Re-scale observations:
                args in `['Scale', ...]`

            - Rotate observations:
                args in `['Rotate', ...]`
        """

        preprocess_dict: dict[str, tuple[str, type[process.BaseProcessLayer]]] = {
            MOVE: ('.*[Mm][Oo][Vv][Ee].*', process.Move),
            ROTATE: ('.*[Rr][Oo][Tt].*', process.Rotate),
            SCALE: ('.*[Ss][Cc][Aa].*', process.Scale),
        }

        process_list = []
        for key, [pattern, processor] in preprocess_dict.items():
            for given_key in kwargs.keys():
                if re.match(pattern, given_key):
                    if (value := kwargs[given_key]) is None:
                        continue

                    elif value == 'auto':
                        value = self._default_process_para[key]

                    process_list.append(processor(self.args.anntype, value))

        self.processor = process.ProcessModel(process_list)

    def process(self, inputs: list[tf.Tensor],
                preprocess: bool,
                update_paras=True,
                training=None,
                *args, **kwargs) -> list[tf.Tensor]:

        if not type(inputs) in [list, tuple]:
            inputs = [inputs]

        if self.processor is None:
            return inputs

        inputs = self.processor.call(inputs, preprocess,
                                     update_paras, training,
                                     *args, **kwargs)
        return inputs

    def load_weights_from_logDir(self, weights_dir: str):
        all_files = os.listdir(weights_dir)
        weights_files = [f for f in all_files
                         if WEIGHTS_FORMAT + '.' in f]
        weights_files.sort()

        if CHECKPOINT_FILENAME in all_files:
            p = os.path.join(weights_dir, CHECKPOINT_FILENAME)
            epoch = int(np.loadtxt(p)[1])

            weights_files = [f for f in weights_files
                             if f'_epoch{epoch}{WEIGHTS_FORMAT}' in f]

        weights_name = weights_files[-1].split('.index')[0]
        self.load_weights(os.path.join(weights_dir, weights_name))
