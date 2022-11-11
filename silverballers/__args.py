"""
@Author: Conghao Wong
@Date: 2022-06-20 21:41:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-11 13:50:17
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY, Args


class _BaseSilverballersArgs(Args):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

        self._set_default('K', 1)
        self._set_default('K_train', 1)

    @property
    def Kc(self) -> int:
        """
        The number of style channels in `Agent` model.
        """
        return self._arg('Kc', 20, argtype=STATIC)

    @property
    def key_points(self) -> str:
        """
        A list of key time steps to be predicted in the agent model.
        For example, `'0_6_11'`.
        """
        return self._arg('key_points', '0_6_11', argtype=STATIC)

    @property
    def preprocess(self) -> str:
        """
        Controls if running any preprocess before model inference.
        Accept a 3-bit-like string value (like `'111'`):
        - The first bit: `MOVE` trajectories to (0, 0);
        - The second bit: re-`SCALE` trajectories;
        - The third bit: `ROTATE` trajectories.
        """
        return self._arg('preprocess', '111', argtype=STATIC)

    @property
    def T(self) -> str:
        """
        Type of transformations used when encoding or decoding
        trajectories.
        It could be:
        - `none`: no transformations
        - `fft`: fast Fourier transform
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('T', 'fft', argtype=STATIC)

    @property
    def feature_dim(self) -> int:
        """
        Feature dimensions that are used in most layers.
        """
        return self._arg('feature_dim', 128, argtype=STATIC)


class AgentArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def depth(self) -> int:
        """
        Depth of the random noise vector.
        """
        return self._arg('depth', 16, argtype=STATIC)


class HandlerArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

        self._set_default('key_points', 'null')

    @property
    def points(self) -> int:
        """
        Controls the number of keypoints accepted in the handler model.
        """
        return self._arg('points', 1, argtype=STATIC)


class SilverballersArgs(_BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def loada(self) -> str:
        """
        Path for agent model.
        """
        return self._arg('loada', 'null', argtype=TEMPORARY)

    @property
    def loadb(self) -> str:
        """
        Path for handler model.
        """
        return self._arg('loadb', 'null', argtype=TEMPORARY)
