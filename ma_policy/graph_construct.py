import numpy as np
import tensorflow as tf
from collections import OrderedDict
from copy import deepcopy
import logging
import traceback
import sys
from ma_policy.variable_schema import VariableSchema, BATCH, TIMESTEPS
from ma_policy.util import shape_list
from ma_policy.layers import (entity_avg_pooling_masked, entity_max_pooling_masked,
                              entity_concat, concat_entity_masks, residual_sa_block,
                              circ_conv1d)

logger = logging.getLogger(__name__)


def construct_tf_graph(all_inputs, spec, act, scope='', reuse=False,):
    '''
        Construct tensorflow graph from spec.
        Args:
            main_inp (tf) -- input activations
            other_inp (dict of tf) -- other input activations such as state
            spec (list of dicts) -- network specification. see Usage below
            scope (string) -- tf variable scope
            reuse (bool) -- tensorflow reuse flag
        Usage:
            Each layer spec has optional arguments: nodes_in and nodes_in. If these arguments
                are omitted, then the default in and out nodes will be 'main'. For layers such as
                concatentation, these arguments must be specified.
            Dense layer (MLP) --
            {
                'layer_type': 'dense'
                'units': int (number of neurons)
                'activation': 'relu', 'tanh', or '' for no activation
            }
            LSTM layer --
            {
                'layer_type': 'lstm'
                'units': int (hidden state size)
            }
            Concat layer --
            Two use cases.
                First: the first input has one less dimension than the second input. In this case,
                    broadcast the first input along the second to last dimension and concatenated
                    along last dimension
                Second: Both inputs have the same dimension, and will be concatenated along last
                    dimension
            {
                'layer_type': 'concat'
                'nodes_in': ['node_one', 'node_two']
                'nodes_out': ['node_out']
            }
            Entity Concat Layer --
            Concatenate along entity dimension (second to last)
            {
                'layer_type': 'entity_concat'
                'nodes_in': ['node_one', 'node_two']
                'nodes_out': ['node_out']
            }
            Entity Self Attention --
            Self attention over entity dimension (second to last)
            See policy.utils:residual_sa_block for args
            {
                'layer_type': 'residual_sa_block'
                'nodes_in': ['node_one']
                'nodes_out': ['node_out']
                ...
            }
            Entity Pooling --
            Pooling along entity dimension (second to last)
            {
                'layer_type': 'entity_pooling'
                'nodes_in': ['node_one', 'node_two']
                'nodes_out': ['node_out']
                'type': (optional string, default 'avg_pooling') type of pooling
                         Current options are 'avg_pooling' and 'max_pooling'
            }
            Circular 1d convolution layer (second to last dimension) --
            {
                'layer_type': 'circ_conv1d',
                'filters': number of filters
                'kernel_size': kernel size
                'activation': 'relu', 'tanh', or '' for no activation
            }
            Flatten outer dimension --
            Flatten all dimensions higher or equal to 3 (necessary after conv layer)
            {
                'layer_type': 'flatten_outer',
            }
            Layernorm --

    '''
    # Make a new dict to not overwrite input
    inp = {k: v for k, v in all_inputs.items()}
    inp['main'] = inp['observation_self']

    valid_activations = {'relu': tf.nn.relu, 'tanh': tf.tanh, '': None}
    state_variables = OrderedDict()
    logger.info(f"Spec:\n{spec}")
    entity_locations = {}
    reset_ops = []
    with tf.variable_scope(scope, reuse=reuse):
        for i, layer in enumerate(spec):
            try:
                layer = deepcopy(layer)
                layer_type = layer.pop('layer_type')
                extra_layer_scope = layer.pop('scope', '')
                nodes_in = layer.pop('nodes_in', ['main'])
                nodes_out = layer.pop('nodes_out', ['main'])
                with tf.variable_scope(extra_layer_scope, reuse=reuse):
                    if layer_type == 'dense':
                        assert len(nodes_in) == len(nodes_out), f"Dense layer must have same number of nodes in as nodes out. \
                            Nodes in: {nodes_in}, Nodes out {nodes_out}"

                        layer['activation'] = valid_activations[layer['activation']]
                        layer_name = layer.pop('layer_name', f'dense{i}')
                        for j in range(len(nodes_in)):
                            inp[nodes_out[j]] = tf.layers.dense(inp[nodes_in[j]],
                                                                name=f'{layer_name}-{j}',
                                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                                reuse=reuse,
                                                                **layer)
                    elif layer_type == 'lstm':
                        layer_name = layer.pop('layer_name', f'lstm{i}')
                        with tf.variable_scope(layer_name, reuse=reuse):
                            assert len(nodes_in) == len(nodes_out) == 1
                            cell = tf.contrib.rnn.BasicLSTMCell(layer['units'])
                            initial_state = tf.contrib.rnn.LSTMStateTuple(inp[scope + f'_lstm{i}_state_c'],
                                                                          inp[scope + f'_lstm{i}_state_h'])
                            inp[nodes_out[0]], state_out = tf.nn.dynamic_rnn(cell,
                                                                             inp[nodes_in[0]],
                                                                             initial_state=initial_state)
                            state_variables[scope + f'_lstm{i}_state_c'] = state_out.c
                            state_variables[scope + f'_lstm{i}_state_h'] = state_out.h
                    elif layer_type == 'concat':
                        layer_name = layer.pop('layer_name', f'concat{i}')
                        with tf.variable_scope(layer_name):
                            assert len(nodes_out) == 1, f"Concat op must only have one node out. Nodes Out: {nodes_out}"
                            assert len(nodes_in) == 2, f"Concat op must have two nodes in. Nodes In: {nodes_in}"
                            assert (len(shape_list(inp[nodes_in[0]])) == len(shape_list(inp[nodes_in[1]])) or
                                    len(shape_list(inp[nodes_in[0]])) == len(shape_list(inp[nodes_in[1]])) - 1),\
                                f"shapes were {nodes_in[0]}:{shape_list(inp[nodes_in[0]])}, {nodes_in[1]}:{shape_list(inp[nodes_in[1]])}"

                            inp0, inp1 = inp[nodes_in[0]], inp[nodes_in[1]]
                            # tile inp0 along second to last dimension to match inp1
                            if len(shape_list(inp[nodes_in[0]])) == len(shape_list(inp1)) - 1:
                                inp0 = tf.expand_dims(inp[nodes_in[0]], -2)
                                tile_dims = [1 for i in range(len(shape_list(inp0)))]
                                tile_dims[-2] = shape_list(inp1)[-2]
                                inp0 = tf.tile(inp0, tile_dims)
                            inp[nodes_out[0]] = tf.concat([inp0, inp1], -1)
                    elif layer_type == 'entity_concat':
                        layer_name = layer.pop('layer_name', f'entity-concat{i}')
                        with tf.variable_scope(layer_name):
                            ec_inps = [inp[node_in] for node_in in nodes_in]
                            inp[nodes_out[0]] = entity_concat(ec_inps)
                            if "masks_in" in layer:
                                masks_in = [inp[_m] if _m is not None else None for _m in layer["masks_in"]]
                                inp[layer["mask_out"]] = concat_entity_masks(ec_inps, masks_in)
                            # Store where the entities are. We'll store with key nodes_out[0]
                            _ent_locs = {}
                            loc = 0
                            for node_in in nodes_in:
                                shape_in = shape_list(inp[node_in])
                                n_ent = shape_in[2] if len(shape_in) == 4 else 1
                                _ent_locs[node_in] = slice(loc, loc + n_ent)
                                loc += n_ent
                            entity_locations[nodes_out[0]] = _ent_locs
                    elif layer_type == 'residual_sa_block':
                        layer_name = layer.pop('layer_name', f'self-attention{i}')
                        with tf.variable_scope(layer_name):
                            assert len(nodes_in) == 1, "self attention should only have one input"
                            sa_inp = inp[nodes_in[0]]

                            mask = inp[layer.pop('mask')] if 'mask' in layer else None
                            internal_layer_name = layer.pop('internal_layer_name', f'residual_sa_block{i}')
                            inp[nodes_out[0]] = residual_sa_block(sa_inp, mask, **layer,
                                                                  scope=internal_layer_name,
                                                                  reuse=reuse)
                    elif layer_type == 'entity_pooling':
                        pool_type = layer.get('type', 'avg_pooling')
                        assert pool_type in ['avg_pooling', 'max_pooling'], f"Pooling type {pool_type} \
                            not available. Pooling type must be either 'avg_pooling' or 'max_pooling'."
                        layer_name = layer.pop('layer_name', f'entity-{pool_type}-pooling{i}')
                        with tf.variable_scope(layer_name):
                            if 'mask' in layer:
                                mask = inp[layer.pop('mask')]
                                assert mask.get_shape()[-1] == inp[nodes_in[0]].get_shape()[-2], \
                                    f"Outer dim of mask must match second to last dim of input. \
                                     Mask shape: {mask.get_shape()}. Input shape: {inp[nodes_in[0]].get_shape()}"
                                if pool_type == 'avg_pooling':
                                    inp[nodes_out[0]] = entity_avg_pooling_masked(inp[nodes_in[0]], mask)
                                elif pool_type == 'max_pooling':
                                    inp[nodes_out[0]] = entity_max_pooling_masked(inp[nodes_in[0]], mask)
                            else:
                                if pool_type == 'avg_pooling':
                                    inp[nodes_out[0]] = tf.reduce_mean(inp[nodes_in[0]], -2)
                                elif pool_type == 'max_pooling':
                                    inp[nodes_out[0]] = tf.reduce_max(inp[nodes_in[0]], -2)
                    elif layer_type == 'circ_conv1d':
                        assert len(nodes_in) == len(nodes_out) == 1, f"Circular convolution layer must have one nodes and one nodes out. \
                            Nodes in: {nodes_in}, Nodes out {nodes_out}"
                        layer_name = layer.pop('layer_name', f'circ_conv1d{i}')
                        with tf.variable_scope(layer_name, reuse=reuse):
                            inp[nodes_out[0]] = circ_conv1d(inp[nodes_in[0]], **layer)
                    elif layer_type == 'flatten_outer':
                        layer_name = layer.pop('layer_name', f'flatten_outer{i}')
                        with tf.variable_scope(layer_name, reuse=reuse):
                            # flatten all dimensions higher or equal to 3
                            inp0 = inp[nodes_in[0]]
                            inp0_shape = shape_list(inp0)
                            inp[nodes_out[0]] = tf.reshape(inp0, shape=inp0_shape[0:2] + [np.prod(inp0_shape[2:])])
                    elif layer_type == "layernorm":
                        layer_name = layer.pop('layer_name', f'layernorm{i}')
                        with tf.variable_scope(layer_name, reuse=reuse):
                            inp[nodes_out[0]] = tf.contrib.layers.layer_norm(inp[nodes_in[0]], begin_norm_axis=2)
                    else:
                        raise NotImplementedError(f"Layer type -- {layer_type} -- not yet implemented")
            except Exception:
                traceback.print_exc(file=sys.stdout)
                print(f"Error in {layer_type} layer: \n{layer}\nNodes in: {nodes_in}, Nodes out: {nodes_out}")
                sys.exit()

    return inp, state_variables, reset_ops


def construct_schemas_zero_state(spec, ob_space, scope=''):
    '''
        Takes a network spec (as specified in construct_tf_graph docstring) and returns
            input schemas and zero states.
    '''
    schemas = OrderedDict()
    zero_states = OrderedDict()
    for i, layer in enumerate(spec):
        layer = deepcopy(layer)
        layer_type = layer.pop('layer_type')

        if layer_type == 'lstm':
            size = tf.contrib.rnn.BasicLSTMCell(layer['units']).state_size
            schemas[scope + f'_lstm{i}_state_c'] = VariableSchema(shape=[BATCH, size.c], dtype=tf.float32)
            schemas[scope + f'_lstm{i}_state_h'] = VariableSchema(shape=[BATCH, size.h], dtype=tf.float32)
            zero_states[scope + f'_lstm{i}_state_c'] = np.expand_dims(np.zeros(size.c, dtype=np.float32), 0)
            zero_states[scope + f'_lstm{i}_state_h'] = np.expand_dims(np.zeros(size.h, dtype=np.float32), 0)

    return schemas, zero_states
