import tensorflow as tf
import numpy as np
import gym
import logging
import sys
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from baselines.common.distributions import make_pdtype

from ma_policy.util import listdict2dictnp, normc_initializer, shape_list, l2_loss
from ma_policy.variable_schema import VariableSchema, BATCH, TIMESTEPS
from ma_policy.normalizers import EMAMeanStd
from ma_policy.graph_construct import construct_tf_graph, construct_schemas_zero_state


class MAPolicy(object):
    '''
        Args:
            ob_space: gym observation space of a SINGLE agent. Expects a dict space.
            ac_space: gym action space. Expects a dict space where each item is a tuple of action
                spaces
            network_spec: list of layers. See construct_tf_graph for details.
            v_network_spec: optional. If specified it is the network spec of the value function.
            trainable_vars: optional. List of variable name segments that should be trained.
            not_trainable_vars: optional. List of variable name segements that should not be
                trained. trainable_vars supercedes this if both are specified.
    '''
    def __init__(self, scope, *, ob_space, ac_space, network_spec, v_network_spec=None,
                 stochastic=True, reuse=False, build_act=True,
                 trainable_vars=None, not_trainable_vars=None,
                 gaussian_fixed_var=True, weight_decay=0.0, ema_beta=0.99999,
                 **kwargs):
        self.reuse = reuse
        self.scope = scope
        self.ob_space = ob_space
        self.ac_space = deepcopy(ac_space)
        self.network_spec = network_spec
        self.v_network_spec = v_network_spec or self.network_spec
        self.stochastic = stochastic
        self.trainable_vars = trainable_vars
        self.not_trainable_vars = not_trainable_vars
        self.gaussian_fixed_var = gaussian_fixed_var
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.build_act = build_act
        self._reset_ops = []
        self._auxiliary_losses = []
        self._running_mean_stds = {}
        self._ema_beta = ema_beta
        self.training_stats = []

        assert isinstance(self.ac_space, gym.spaces.Dict)
        assert isinstance(self.ob_space, gym.spaces.Dict)
        assert 'observation_self' in self.ob_space.spaces

        # Action space will come in as a MA action space. Convert to a SA action space.
        self.ac_space.spaces = {k: v.spaces[0] for k, v in self.ac_space.spaces.items()}

        self.pdtypes = {k: make_pdtype(s) for k, s in self.ac_space.spaces.items()}

        # Create input schemas for each action type
        self.input_schemas = {
            k: VariableSchema(shape=[BATCH, TIMESTEPS] + pdtype.sample_shape(),
                              dtype=pdtype.sample_dtype())
            for k, pdtype in self.pdtypes.items()
        }

        # Creat input schemas for each observation
        for k, v in self.ob_space.spaces.items():
            self.input_schemas[k] = VariableSchema(shape=[BATCH, TIMESTEPS] + list(v.shape),
                                                   dtype=tf.float32)

        # Setup schemas and zero state for layers with state
        v_state_schemas, v_zero_states = construct_schemas_zero_state(
            self.v_network_spec, self.ob_space, 'vpred_net')
        pi_state_schemas, pi_zero_states = construct_schemas_zero_state(
            self.network_spec, self.ob_space, 'policy_net')

        self.state_keys = list(v_state_schemas.keys()) + list(pi_state_schemas.keys())
        self.input_schemas.update(v_state_schemas)
        self.input_schemas.update(pi_state_schemas)
        self.zero_state = {}
        self.zero_state.update(v_zero_states)
        self.zero_state.update(pi_zero_states)

        if build_act:
            with tf.variable_scope(self.scope, reuse=self.reuse):
                self.phs = {name: schema.placeholder(name=name)
                            for name, schema in self.get_input_schemas().items()}
            self.build(self.phs)

    def build(self, inputs):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.full_scope_name = tf.get_variable_scope().name
            self._init(inputs, **self.kwargs)

    def _init(self, inputs, gaussian_fixed_var=True, **kwargs):
        '''
            Args:
                inputs (dict): input dictionary containing tf tensors
                gaussian_fixed_var (bool): If True the policies variance won't be conditioned on state
        '''
        taken_actions = {k: inputs[k] for k in self.pdtypes.keys()}

        #  Copy inputs to not overwrite. Don't need to pass actions to policy, so exlcude these
        processed_inp = {k: v for k, v in inputs.items() if k not in self.pdtypes.keys()}

        self._normalize_inputs(processed_inp)

        self.state_out = OrderedDict()

        # Value network
        (vpred,
         vpred_state_out,
         vpred_reset_ops) = construct_tf_graph(
            processed_inp, self.v_network_spec, scope='vpred_net', act=self.build_act)

        self._init_vpred_head(vpred, processed_inp, 'vpred_out0', "value0")

        # Policy network
        (pi,
         pi_state_out,
         pi_reset_ops) = construct_tf_graph(
            processed_inp, self.network_spec, scope='policy_net', act=self.build_act)

        self.state_out.update(vpred_state_out)
        self.state_out.update(pi_state_out)
        self._reset_ops += vpred_reset_ops + pi_reset_ops
        self._init_policy_out(pi, taken_actions)
        if self.weight_decay != 0.0:
            kernels = [var for var in self.get_trainable_variables() if 'kernel' in var.name]
            w_norm_sum = tf.reduce_sum([tf.nn.l2_loss(var) for var in kernels])
            w_norm_loss = w_norm_sum * self.weight_decay
            self.add_auxiliary_loss('weight_decay', w_norm_loss)

        # set state to zero state
        self.reset()

    def _init_policy_out(self, pi, taken_actions):
        with tf.variable_scope('policy_out'):
            self.pdparams = {}
            for k in self.pdtypes.keys():
                with tf.variable_scope(k):
                    if self.gaussian_fixed_var and isinstance(self.ac_space.spaces[k], gym.spaces.Box):
                        mean = tf.layers.dense(pi["main"],
                                               self.pdtypes[k].param_shape()[0] // 2,
                                               kernel_initializer=normc_initializer(0.01),
                                               activation=None)
                        logstd = tf.get_variable(name="logstd",
                                                 shape=[1, self.pdtypes[k].param_shape()[0] // 2],
                                                 initializer=tf.zeros_initializer())
                        self.pdparams[k] = tf.concat([mean, mean * 0.0 + logstd], axis=2)
                    elif k in pi:
                        # This is just for the case of entity specific actions
                        if isinstance(self.ac_space.spaces[k], (gym.spaces.Discrete)):
                            assert pi[k].get_shape()[-1] == 1
                            self.pdparams[k] = pi[k][..., 0]
                        elif isinstance(self.ac_space.spaces[k], (gym.spaces.MultiDiscrete)):
                            assert np.prod(pi[k].get_shape()[-2:]) == self.pdtypes[k].param_shape()[0],\
                                f"policy had shape {pi[k].get_shape()} for action {k}, but required {self.pdtypes[k].param_shape()}"
                            new_shape = shape_list(pi[k])[:-2] + [np.prod(pi[k].get_shape()[-2:]).value]
                            self.pdparams[k] = tf.reshape(pi[k], shape=new_shape)
                        else:
                            assert False
                    else:
                        self.pdparams[k] = tf.layers.dense(pi["main"],
                                                           self.pdtypes[k].param_shape()[0],
                                                           kernel_initializer=normc_initializer(0.01),
                                                           activation=None)

            with tf.variable_scope('pds'):
                self.pds = {k: pdtype.pdfromflat(self.pdparams[k])
                            for k, pdtype in self.pdtypes.items()}

            with tf.variable_scope('sampled_action'):
                self.sampled_action = {k: pd.sample() if self.stochastic else pd.mode()
                                       for k, pd in self.pds.items()}
            with tf.variable_scope('sampled_action_logp'):
                self.sampled_action_logp = sum([self.pds[k].logp(self.sampled_action[k])
                                                for k in self.pdtypes.keys()])
            with tf.variable_scope('entropy'):
                self.entropy = sum([pd.entropy() for pd in self.pds.values()])
            with tf.variable_scope('taken_action_logp'):
                self.taken_action_logp = sum([self.pds[k].logp(taken_actions[k])
                                              for k in self.pdtypes.keys()])

    def _init_vpred_head(self, vpred, processed_inp, vpred_scope, feedback_name):
        with tf.variable_scope(vpred_scope):
            _vpred = tf.layers.dense(vpred['main'], 1, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            _vpred = tf.squeeze(_vpred, -1)
            normalize_axes = (0, 1)
            loss_fn = partial(l2_loss, mask=processed_inp.get(feedback_name + "_mask", None))
            rms_class = partial(EMAMeanStd, beta=self._ema_beta)

            rms_shape = [dim for i, dim in enumerate(_vpred.get_shape()) if i not in normalize_axes]
            self.value_rms = rms_class(shape=rms_shape, scope='value0filter')
            self.scaled_value_tensor = self.value_rms.mean + _vpred * self.value_rms.std
            self.add_running_mean_std(rms=self.value_rms, name='feedback.value0', axes=normalize_axes)

    def _normalize_inputs(self, processed_inp):
        with tf.variable_scope('normalize_self_obs'):
            ob_rms_self = EMAMeanStd(shape=self.ob_space.spaces['observation_self'].shape,
                                     scope="obsfilter", beta=self._ema_beta, per_element_update=False)
            self.add_running_mean_std("observation_self", ob_rms_self, axes=(0, 1))
            normalized = (processed_inp['observation_self'] - ob_rms_self.mean) / ob_rms_self.std
            clipped = tf.clip_by_value(normalized, -5.0, 5.0)
            processed_inp['observation_self'] = clipped

        for key in self.ob_space.spaces.keys():
            if key == 'observation_self':
                continue
            elif 'mask' in key:  # Don't normalize observation masks
                pass
            else:
                with tf.variable_scope(f'normalize_{key}'):
                    ob_rms = EMAMeanStd(shape=self.ob_space.spaces[key].shape[1:],
                                        scope=f"obsfilter/{key}", beta=self._ema_beta, per_element_update=False)
                    normalized = (processed_inp[key] - ob_rms.mean) / ob_rms.std
                    processed_inp[key] = tf.clip_by_value(normalized, -5.0, 5.0)
                    self.add_running_mean_std(key, ob_rms, axes=(0, 1, 2))

    def get_input_schemas(self):
        return self.input_schemas.copy()

    def process_state_batch(self, states):
        '''
            Batch states together.
            args:
                states -- list (batch) of dicts of states with shape (n_agent, dim state).
        '''
        new_states = listdict2dictnp(states, keepdims=True)
        return new_states

    def process_observation_batch(self, obs):
        '''
            Batch obs together.
            Args:
                obs -- list of lists (batch, time), where elements are dictionary observations
        '''

        new_obs = deepcopy(obs)
        # List tranpose -- now in (time, batch)
        new_obs = list(map(list, zip(*new_obs)))
        # Convert list of list of dicts to dict of numpy arrays
        new_obs = listdict2dictnp([listdict2dictnp(batch, keepdims=True) for batch in new_obs])
        # Flatten out the agent dimension, so batches look like normal SA batches
        new_obs = {k: self.reshape_ma_observations(v) for k, v in new_obs.items()}

        return new_obs

    def reshape_ma_observations(self, obs):
        # Observations with shape (time, batch)
        if len(obs.shape) == 2:
            batch_first_ordering = (1, 0)
        # Observations with shape (time, batch, dim obs)
        elif len(obs.shape) == 3:
            batch_first_ordering = (1, 0, 2)
        # Observations with shape (time, batch, n_entity, dim obs)
        elif len(obs.shape) == 4:
            batch_first_ordering = (1, 0, 2, 3)
        else:
            raise ValueError(f"Obs dim {obs.shape}. Only supports dim 3 or 4")
        new_obs = obs.copy().transpose(batch_first_ordering)  # (n_agent, batch, time, dim obs)

        return new_obs

    def prepare_input(self, observation, state_in, taken_action=None):
        ''' Add in time dimension to observations, assumes that first dimension of observation is
            already the batch dimension and does not need to be added.'''
        obs = deepcopy(observation)
        obs.update(state_in)
        if taken_action is not None:
            obs.update(taken_action)
        return obs

    def act(self, observation, extra_feed_dict={}):
        outputs = {
            'ac': self.sampled_action,
            'ac_logp': self.sampled_action_logp,
            'vpred': self.scaled_value_tensor,
            'state': self.state_out}
        # Add timestep dimension to observations
        obs = deepcopy(observation)
        n_agents = observation['observation_self'].shape[0]

        # Make sure that there are as many states as there are agents.
        # This should only happen with the zero state.
        for k, v in self.state.items():
            assert v.shape[0] == 1 or v.shape[0] == n_agents
            if v.shape[0] == 1 and v.shape[0] != n_agents:
                self.state[k] = np.repeat(v, n_agents, 0)

        # Add time dimension to obs
        for k, v in obs.items():
            obs[k] = np.expand_dims(v, 1)
        inputs = self.prepare_input(observation=obs, state_in=self.state)
        feed_dict = {self.phs[k]: v for k, v in inputs.items()}
        feed_dict.update(extra_feed_dict)

        outputs = tf.get_default_session().run(outputs, feed_dict)
        self.state = outputs['state']

        # Remove time dimension from outputs
        def preprocess_act_output(act_output):
            if isinstance(act_output, dict):
                return {k: np.squeeze(v, 1) for k, v in act_output.items()}
            else:
                return np.squeeze(act_output, 1)

        info = {'vpred': preprocess_act_output(outputs['vpred']),
                'ac_logp': preprocess_act_output(outputs['ac_logp']),
                'state': outputs['state']}

        return preprocess_act_output(outputs['ac']), info

    def get_variables(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.full_scope_name + '/')
        return variables

    def get_trainable_variables(self):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.full_scope_name + '/')
        if self.trainable_vars is not None:
            variables = [v for v in variables
                         if any([tr_v in v.name for tr_v in self.trainable_vars])]
        elif self.not_trainable_vars is not None:
            variables = [v for v in variables
                         if not any([tr_v in v.name for tr_v in self.not_trainable_vars])]
        variables = [v for v in variables if 'not_trainable' not in v.name]
        return variables

    def reset(self):
        self.state = deepcopy(self.zero_state)
        if tf.get_default_session() is not None:
            tf.get_default_session().run(self._reset_ops)

    def set_state(self, state):
        self.state = deepcopy(state)

    def auxiliary_losses(self):
        """ Any extra losses internal to the policy, automatically added to the total loss."""
        return self._auxiliary_losses

    def add_auxiliary_loss(self, name, loss):
        self.training_stats.append((name, 'scalar', loss, lambda x: x))
        self._auxiliary_losses.append(loss)

    def add_running_mean_std(self, name, rms, axes=(0, 1)):
        """
        Add a RunningMeanStd/EMAMeanStd object to the policy's list. It will then get updated during optimization.
        :param name: name of the input field to update from.
        :param rms: RMS object to update.
        :param axes: axes of the input to average over.
            RMS's shape should be equal to input's shape after axes are removed.
            e.g. if inputs is [5, 6, 7, 8] and axes is [0, 2], then RMS's shape should be [6, 8].
        :return:
        """
        self._running_mean_stds[name] = {'rms': rms, 'axes': axes}
