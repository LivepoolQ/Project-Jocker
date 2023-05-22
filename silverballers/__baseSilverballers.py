"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-22 20:42:50
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.base import BaseObject
from codes.constant import ANN_TYPES, INPUT_TYPES
from codes.managers import AnnotationManager, Model, Structure

from .__args import AgentArgs, SilverballersArgs
from .agents import BaseAgentModel, BaseAgentStructure
from .handlers import BaseHandlerModel, BaseHandlerStructure


class BaseSilverballersModel(Model):
    """
    BaseSilverballersModel
    ---
    The two-stage silverballers model.
    NOTE: This model is typically used for testing, not training.

    Member Managers
    ---
    - (Soft member) Stage-1 Subnetwork, type is `BaseAgentModel`
        or a subclass of it;
    - (Soft member) Stage-2 Subnetwork, type is `BaseHandlerModel`
        or a subclass of it.
    """

    def __init__(self, Args: SilverballersArgs,
                 agentModel: BaseAgentModel,
                 handlerModel: BaseHandlerModel = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args: SilverballersArgs
        self.manager: Structure

        # processes are run in AgentModels and HandlerModels
        self.set_preprocess()

        # Layers
        self.agent = agentModel
        self.handler = handlerModel

        # Set model inputs
        a_type = self.agent.input_types
        h_type = self.handler.input_types[:-1]
        self.input_types = list(set(a_type + h_type))
        self.agent_input_index = self.get_input_index(a_type)
        self.handler_input_index = self.get_input_index(h_type)

    def get_input_index(self, input_type: list[str]):
        return [self.input_types.index(t) for t in input_type]

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None,
             *args, **kwargs):

        # Predict with `co2bb` (Coordinates to 2D bounding boxes)
        if self.args.force_anntype == ANN_TYPES.BB_2D and \
           self.agent.args.anntype == ANN_TYPES.CO_2D and \
           self.manager.split_manager.anntype == ANN_TYPES.BB_2D:

            # Flatten into a series of 2D points
            all_trajs = self.manager.get_member(AnnotationManager) \
                .target.get_coordinate_series(inputs[0])

        # Predict multiple times along with the time axis
        else:
            all_trajs = [inputs[0]]

        results = []
        while len(all_trajs):
            traj = all_trajs.pop(0)
            inputs_new = (traj,) + inputs[1:]

            # call the first stage model
            agent_inputs = [inputs_new[i] for i in self.agent_input_index]
            agent_proposals = self.agent.forward(agent_inputs)[0]

            # down sampling from K*Kc generations (if needed)
            if self.args.down_sampling_rate < 1.0:
                K_current = agent_proposals.shape[1]
                K_new = K_current * self.args.down_sampling_rate

                new_index = tf.random.shuffle(tf.range(K_current))[:int(K_new)]
                agent_proposals = tf.gather(agent_proposals, new_index, axis=1)

            # call the second stage model
            handler_inputs = [inputs_new[i] for i in self.handler_input_index]
            handler_inputs.append(agent_proposals)
            final_results = self.handler.forward(handler_inputs)[0]

            # process results
            if self.args.pred_frames > self.agent.args.pred_frames:
                i = self.args.pred_interval

                # reshape into (batch, pred, dim)
                final_results = final_results[:, 0, :, :]

                if not len(results):
                    results.append(final_results)
                else:
                    results[0] = tf.concat([results[0],
                                            final_results[:, -i:, :]], axis=-2)

                # check current predictions
                if results[0].shape[-2] >= self.args.pred_frames:
                    results[0] = results[0][:, :self.args.pred_frames, :]
                    break

                # get next observations
                whole_traj = tf.concat([traj, final_results], axis=-2)

                new_obs_traj = whole_traj[:, i:i+self.args.obs_frames, :]
                all_trajs.append(new_obs_traj)

            else:
                results.append(final_results)

        return (tf.concat(results, axis=-1),)

    def print_info(self, **kwargs):
        info = {'Index of keypoints': self.agent.p_index,
                'Stage-1 Subnetwork': f"'{self.agent.name}' from '{self.structure.args.loada}'",
                'Stage-2 Subnetwork': f"'{self.handler.name}' from '{self.structure.args.loadb}'"}

        kwargs_old = kwargs.copy()
        kwargs.update(**info)
        super().print_info(**kwargs)

        self.agent.print_info(**kwargs_old)
        self.handler.print_info(**kwargs_old)


class BaseSilverballers(Structure):
    """
    BaseSilverballers
    ---
    Basic structure to run the `agent-handler` based silverballers model.
    NOTE: It is only used for TESTING silverballers models, not training.
    Please set agent model and handler model used in this silverballers by
    subclassing this class, and call the `set_models` method *before*
    the `super().__init__()` method.

    Member Managers
    ---
    - Stage-1 Subnetwork Manager, type is `BaseAgentStructure` or its subclass;
    - Stage-2 Subnetwork Manager, type is `BaseHandlerStructure` or its subclass;
    - All members from the `Structure`.
    """

    # Structures
    agent_structure = BaseAgentStructure
    handler_structure = BaseHandlerStructure

    # Models
    agent_model = None
    handler_model = None
    silverballer_model = BaseSilverballersModel

    def __init__(self, terminal_args: list[str]):

        # Init log-related functions
        BaseObject.__init__(self)

        # Load minimal args
        min_args = SilverballersArgs(terminal_args, is_temporary=True)

        # Check args
        if 'null' in [min_args.loada, min_args.loadb]:
            self.log('`Agent` or `Handler` model not found!' +
                     ' Please specific their paths via `--loada` (`-la`)' +
                     ' or `--loadb` (`-lb`).',
                     level='error', raiseError=KeyError)

        # Load basic args from the saved agent model
        min_args_a = AgentArgs(terminal_args + ['--load', min_args.loada],
                               is_temporary=True)

        # Assign args from the saved Agent-Model's args
        extra_args = []
        if min_args.batch_size > min_args_a.batch_size:
            extra_args += ['--batch_size', str(min_args_a.batch_size)]

        # Check if predict trajectories recurrently
        if 'pred_frames' not in min_args._args_runnning.keys():
            extra_args += ['--pred_frames', str(min_args_a.pred_frames)]
        else:
            if not min_args_a.deterministic:
                self.log('Predict trajectories currently is currently not ' +
                         'supported with generative models.',
                         level='error', raiseError=NotImplementedError)

            if min_args.pred_interval == -1:
                self.log('Please set the prediction interval when you want ' +
                         'to make recurrent predictions. Current prediction' +
                         f' interval is `{min_args.pred_interval}`.',
                         level='error', raiseError=ValueError)

        extra_args += ['--split', str(min_args_a.split),
                       '--anntype', str(min_args_a.anntype),
                       '--obs_frames', str(min_args_a.obs_frames),
                       '--interval', str(min_args_a.interval)]

        self.args = SilverballersArgs(terminal_args + extra_args)

        if self.args.force_anntype != 'null':
            self.args._set('anntype', self.args.force_anntype)

        # init the structure
        super().__init__(self.args)

        if (k := '--force_anntype') in terminal_args:
            terminal_args.remove(k)

        self.noTraining = True

        # config second-stage model
        if self.handler_model.is_interp_handler:
            handler_args = None
            handler_path = None
        else:
            handler_args = terminal_args + ['--load', self.args.loadb]
            handler_path = self.args.loadb

        # assign substructures
        self.agent = self.substructure(self.agent_structure,
                                       args=(terminal_args +
                                             ['--load', self.args.loada]),
                                       model=self.agent_model,
                                       load=self.args.loada)

        self.handler = self.substructure(self.handler_structure,
                                         args=handler_args,
                                         model=self.handler_model,
                                         create_args=dict(asHandler=True),
                                         load=handler_path,
                                         key_points=self.agent.args.key_points)

        # set labels
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

    def substructure(self, structure: type[BaseAgentStructure],
                     args: list[str],
                     model: type[BaseAgentModel],
                     create_args: dict = {},
                     load: str = None,
                     **kwargs):
        """
        Init a sub-structure (which contains its corresponding model).

        :param structure: class name of the training structure
        :param args: args to init the training structure
        :param model: class name of the model
        :param create_args: args to create the model, and they will be fed
            to the `structure.create_model` method
        :param load: path to load model weights
        :param **kwargs: a series of force-args that will be assigned to
            the structure's args
        """

        struct = structure(args, manager=self, is_temporary=True)
        for key in kwargs.keys():
            struct.args._set(key, kwargs[key])

        struct.set_model_type(model)
        struct.model = struct.create_model(**create_args)

        if load:
            struct.model.load_weights_from_logDir(load)

        return struct

    def set_models(self, agentModel: type[BaseAgentModel],
                   handlerModel: type[BaseHandlerModel],
                   agentStructure: type[BaseAgentStructure] = None,
                   handlerStructure: type[BaseHandlerStructure] = None):
        """
        Set models and structures used in this silverballers instance.
        Please call this method before the `__init__` method when subclassing.
        You should better set `agentModel` and `handlerModel` rather than
        their training structures if you do not subclass these structures.
        """
        if agentModel:
            self.agent_model = agentModel

        if agentStructure:
            self.agent_structure = agentStructure

        if handlerModel:
            self.handler_model = handlerModel

        if handlerStructure:
            self.handler_structure = handlerStructure

    def create_model(self, *args, **kwargs):
        return self.silverballer_model(
            self.args,
            agentModel=self.agent.model,
            handlerModel=self.handler.model,
            structure=self,
            *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with 1st sub-network `{self.args.loada}` ' +
                 f'and 2nd seb-network `{self.args.loadb}` done.')
