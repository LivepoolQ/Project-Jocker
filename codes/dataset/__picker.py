"""
@Author: Conghao Wong
@Date: 2022-08-30 09:52:17
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-19 11:35:05
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ..base import BaseManager

T_2D_COORDINATE = 'coordinate'
T_2D_BOUNDINGBOX = 'boundingbox'
T_2D_BOUNDINGBOX_ROTATE = 'boundingbox-rotate'
T_3D_BOUNDINGBOX = '3Dboundingbox'
T_3D_BOUNDINGBOX_ROTATE = '3Dboundingbox-rotate'


class _BaseAnnType():
    def __init__(self) -> None:
        self.typeName: str = None
        self.dim: int = None
        self.targets: list[type[_BaseAnnType]] = []

    def transfer(self, target, traj: np.ndarray) -> np.ndarray:
        """
        Transfer the n-dim trajectory to the other m-dim trajectory.

        :param target: an instance or subclass of `_BaseAnnType` that \
            manages the m-dim trajectory 
        :param traj: n-dim trajectory
        """
        T = type(target)
        if T == type(self):
            return traj

        if not T in self.targets:
            T_c = self.__class__.__name__
            raise ValueError(f'Transfer from {T_c} to {T} is not supported.')

        else:
            if traj.ndim == 4:       # (batch, steps, K, dim)
                _traj = np.transpose(traj, [3, 0, 1, 2])
            elif traj.ndim == 3:      # (batch, steps, dim)
                _traj = np.transpose(traj, [2, 0, 1])
            elif traj.ndim == 2:    # (steps, dim)
                _traj = traj.T
            else:
                raise NotImplementedError

            # shape of `_traj` is (dim, ...)
            return self._transfer(T, _traj)

    def _transfer(self, target, traj: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Coordinate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = T_2D_COORDINATE
        self.dim = 2
        self.targets = []


class _Boundingbox(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = T_2D_BOUNDINGBOX
        self.dim = 4
        self.targets = [_Coordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        if T == _Coordinate:
            xl, yl, xr, yr = traj[:4]
            return 0.5 * np.stack((xl+xr, yl+yr), axis=-1)

        else:
            raise NotImplementedError


class _3DBoundingboxWithRotate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = T_3D_BOUNDINGBOX_ROTATE
        self.dim = 10
        self.targets = [_Coordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        if T == _Coordinate:
            xl, yl, xr, yr = traj[:4]
            return 0.5 * np.stack((xl+xr, yl+yr), axis=-1)

        else:
            raise NotImplementedError


class Picker():
    """
    Picker
    ---

    Picker object to get trajectories from the n-dim meta-trajectories.
    """

    def __init__(self, datasetType: str, predictionType: str):
        """
        Both argument `datasetType` and `predictionType` accept strings:
        - `'coordinate'`
        - `'boundingbox'`
        - `'boundingbox-rotate'`
        - `'3Dboundingbox'`
        - `'3Dboundingbox-rotate'`

        :param datasetType: type of the dataset annotation files
        :param predictionType: type of the model predictions
        """
        super().__init__()

        self.ds_type = datasetType
        self.pred_type = predictionType

        self.ds_manager = get_manager(datasetType)
        self.pred_manager = get_manager(predictionType)

    def get(self, traj: np.ndarray) -> np.ndarray:
        """
        Get trajectories from the n-dim meta-trajectories.
        """
        return self.ds_manager.transfer(self.pred_manager, traj)


class AnnotationManager(BaseManager):

    def __init__(self, manager: BaseManager,
                 dataset_type: str,
                 name: str = 'Annotation Manager'):

        super().__init__(manager=manager, name=name)

        self.d_type = dataset_type
        self.p_type = self.args.anntype

        self.dataset_picker = Picker(datasetType=dataset_type,
                                     predictionType=self.p_type)

        self.center_picker = Picker(datasetType=self.p_type,
                                    predictionType=T_2D_COORDINATE)

    def get(self, inputs: np.ndarray):
        """
        Get data with target annotations from original dataset files.
        """
        return self.dataset_picker.get(inputs)

    def get_center(self, inputs: np.ndarray):
        """
        Get the center of trajectories from the processed data.
        Note that annotation type of `inputs` is the same as model's
        prediction type. (Not the dataset's annotation type.)
        """
        return self.center_picker.get(inputs)

    def print_info(self, **kwargs):
        info = {'Dataset annotation type': self.d_type,
                'Model prediction type': self.p_type}

        kwargs.update(**info)
        return super().print_info(**kwargs)


def get_manager(anntype: str) -> _BaseAnnType:
    if anntype == T_2D_COORDINATE:
        return _Coordinate()
    elif anntype == T_2D_BOUNDINGBOX:
        return _Boundingbox()
    elif anntype == T_2D_BOUNDINGBOX_ROTATE:
        raise NotImplementedError(anntype)
    elif anntype == T_3D_BOUNDINGBOX:
        raise NotImplementedError(anntype)
    elif anntype == T_3D_BOUNDINGBOX_ROTATE:
        return _3DBoundingboxWithRotate()
    else:
        raise NotImplementedError(anntype)
