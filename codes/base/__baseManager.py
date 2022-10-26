"""
@Author: Conghao Wong
@Date: 2022-10-17 14:57:03
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-21 10:15:30
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
from typing import TypeVar, Union

import tensorflow as tf
from tqdm import tqdm

from ..args import Args
from ..utils import MAX_PRINT_LIST_LEN

T = TypeVar('T')


class _BaseManager():

    def __init__(self, name: str = None):
        super().__init__()

        try:
            self.name = name
        except AttributeError:
            pass

        # create or restore a logger
        logger = logging.getLogger(name=f'`{name}` ({type(self).__name__})')

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)

            # add file handler
            fhandler = logging.FileHandler(filename='./test.log', mode='a')
            fhandler.setLevel(logging.INFO)

            # add terminal handler
            thandler = logging.StreamHandler()
            thandler.setLevel(logging.INFO)

            # add formatter
            fformatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s] %(name)s: %(message)s')
            fhandler.setFormatter(fformatter)

            tformatter = logging.Formatter(
                '[%(levelname)s] %(name)s: %(message)s')
            thandler.setFormatter(tformatter)

            logger.addHandler(fhandler)
            logger.addHandler(thandler)

        self.logger = logger
        self.bar: tqdm = None

    def log(self, s: str, level: str = 'info'):
        """
        Log infomation to files and console

        :param s: text to log
        :param level: log level, canbe `'info'` or `'error'` or `'debug'`
        """
        if level == 'info':
            self.logger.info(s)

        elif level == 'error':
            self.logger.error(s)

        elif level == 'debug':
            self.logger.debug(s)

        else:
            raise NotImplementedError

        return s

    def timebar(self, inputs: T, text='') -> T:
        self.bar = tqdm(inputs, desc=text)
        return self.bar

    def update_timebar(self, item: Union[str, dict], pos='end'):
        """
        Update the tqdm timebar.

        :param item: string or dict to update
        :param pos: position, canbe `'end'` or `'start'`
        """
        if pos == 'end':
            if type(item) is str:
                self.bar.set_postfix_str(item)
            elif type(item) is dict:
                self.bar.set_postfix(item)
            else:
                raise ValueError(item)

        elif pos == 'start':
            self.bar.set_description(item)
        else:
            raise NotImplementedError(pos)

    def print_info(self, **kwargs):
        """
        Print information of the object itself.
        """
        self.print_parameters(**kwargs)

    def print_parameters(self, title='null', **kwargs):
        if title == 'null':
            title = ''

        print(f'>>> [{self.name}]: {title}')
        for key, value in kwargs.items():
            if type(value) == tf.Tensor:
                value = value.numpy()

            if (type(value) == list and
                    len(value) > MAX_PRINT_LIST_LEN):
                value = value[:MAX_PRINT_LIST_LEN] + ['...']

            print(f'    - {key}: {value}.')

        print('')

    @staticmethod
    def log_bar(percent, total_length=30):

        bar = (''.join('=' * (int(percent * total_length) - 1))
               + '>')
        return bar


# It is used for type-hinting
class BaseManager(_BaseManager):
    """
    BaseManager
    ----------
    Base class for all structures.

    Public Methods
    --------------
    ### Manager and members methods
    ```python
    # get a member by type
    (method) get_member: (self: Self@BaseManager,
                          mtype: Type[T@get_member],
                          mindex: int = 0) -> T@get_member

    # get all members with the same type
    (method) find_members_by_type: (self: Self@BaseManager,
                                    mtype: Type[T@find_members_by_type]) \
                                         -> list[T@find_members_by_type]
    ```

    ### Log and print methods
    ```
    # log information
    (method) log: (self: Self@BaseManager, s: str, level: str = 'info') -> str

    # print parameters with the format
    (method) print_parameters: (self: Self@BaseManager, 
                                title: str = 'null',
                                **kwargs: Any) -> None

    # print information of the manager object itself
    (method) print_info: (self: Self@BaseManager, **kwargs: Any) -> None

    # print information of the manager and its members
    (method) print_info_all: (self: Self@BaseManager,
                              include_self: bool = True) -> None

    # timebar
    (method) log_timebar: (inputs, text='', return_enumerate=True) -> (enumerate | tqdm)
    ```
    """

    def __init__(self, args: Args = None,
                 manager: _BaseManager = None,
                 name: str = None):

        super().__init__(name)
        self._args: Args = args
        self.manager: BaseManager = manager
        self.members: list[BaseManager] = []
        self.members_dict: dict[type[BaseManager], list[BaseManager]] = {}

        if manager:
            mtype = type(self)
            if not mtype in self.manager.members_dict.keys():
                self.manager.members_dict[mtype] = []

            self.manager.members_dict[mtype].append(self)
            self.manager.members.append(self)

    @property
    def args(self) -> Args:
        if self._args:
            return self._args
        elif self.manager:
            return self.manager.args
        else:
            return None

    @args.setter
    def args(self, value: T) -> T:
        self._args = value

    def get_member(self, mtype: type[T], mindex: int = 0) -> T:
        """
        Get a member manager by class name.

        :param mtype: Type of the member manager.
        :param mindex: Index of the member.

        :return member: Member manager with the specific type.
        """
        return self.members_dict[mtype][mindex]

    def find_members_by_type(self, mtype: type[T]) -> list[T]:
        """
        Find member managers by class name.

        :param mtype: Type of the member manager.

        :return members: A list of member objects.
        """
        return self.members_dict[mtype]

    def print_info_all(self, include_self=True):
        """
        Print information of the object itself and all its members.
        It is used to debug only.
        """
        if include_self:
            self.print_info(title='DEBUG', object=self, members=self.members)

        for s in self.members:
            s.print_info(title='DEBUG', object=s,
                         manager=self, members=s.members)
            s.print_info_all(include_self=False)