import numpy as np
import tensorflow as tf
from ma_policy.util import shape_list


#################
# Pooling #######
#################

def entity_avg_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    masked = x * mask
    summed = tf.reduce_sum(masked, -2)
    denom = tf.reduce_sum(mask, -2) + 1e-5
    return summed / denom


def entity_max_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    has_unmasked_entities = tf.sign(tf.reduce_sum(mask, axis=-2, keepdims=True))
    offset = (mask - 1) * 1e9
    masked = (x + offset) * has_unmasked_entities
    return tf.reduce_max(masked, -2)


#################
# Contat Ops ####
#################

def entity_concat(inps):
    '''
        Concat 4D tensors along the third dimension. If a 3D tensor is in the list
            then treat it as a single entity and expand the third dimension
        Args:
            inps (list of tensors): tensors to concatenate
    '''
    with tf.variable_scope('concat_entities'):
        shapes = [shape_list(_x) for _x in inps]
        # For inputs that don't have entity dimension add one.
        inps = [_x if len(_shape) == 4 else tf.expand_dims(_x, 2) for _x, _shape in zip(inps, shapes)]
        shapes = [shape_list(_x) for _x in inps]
        assert np.all([_shape[-1] == shapes[0][-1] for _shape in shapes]),\
            f"Some entities don't have the same outer or inner dimensions {shapes}"
        # Concatenate along entity dimension
        out = tf.concat(inps, -2)
    return out


def concat_entity_masks(inps, masks):
    '''
        Concats masks together. If mask is None, then it creates
            a tensor of 1's with shape (BS, T, NE).
        Args:
            inps (list of tensors): tensors that masks apply to
            masks (list of tensors): corresponding masks
    '''
    assert len(inps) == len(masks), "There should be the same number of inputs as masks"
    with tf.variable_scope('concat_masks'):
        shapes = [shape_list(_x) for _x in inps]
        new_masks = []
        for inp, mask in zip(inps, masks):
            if mask is None:
                inp_shape = shape_list(inp)
                if len(inp_shape) == 4:  # this is an entity tensor
                    new_masks.append(tf.ones(inp_shape[:3]))
                elif len(inp_shape) == 3:  # this is a pooled or main tensor. Set NE (outer dimension) to 1
                    new_masks.append(tf.ones(inp_shape[:2] + [1]))
            else:
                new_masks.append(mask)
        new_mask = tf.concat(new_masks, -1)
    return new_mask


#################
# Transformer ###
#################


def residual_sa_block(inp, mask, heads, n_embd,
                      layer_norm=False, post_sa_layer_norm=False,
                      n_mlp=1, qk_w=0.125, v_w=0.125, post_w=0.125,
                      mlp_w1=0.125, mlp_w2=0.125,
                      scope="residual_sa_block", reuse=False):
    '''
        Residual self attention block for entities.
        Notation:
            T  - Time
            NE - Number entities
        Args:
            inp (tf): (BS, T, NE, f)
            mask (tf): (BS, T, NE)
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            n_mlp (int) -- number of mlp layers. If there are more than 1 mlp layers, we'll add a residual
                connection from after the first mlp to after the last mlp.
            qk_w, v_w, post_w, mlp_w1, mlp_w2 (float) -- scale for gaussian init for keys/queries, values, mlp
                post self attention, second mlp, and third mlp, respectively. Std will be sqrt(scale/n_embd)
            scope (string) -- tf scope
            reuse (bool) -- tf reuse
    '''
    with tf.variable_scope(scope, reuse=reuse):
        a = self_attention(inp, mask, heads, n_embd, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w,
                           scope='self_attention', reuse=reuse)
        post_scale = np.sqrt(post_w / n_embd)
        post_a_mlp = tf.layers.dense(a,
                                     n_embd,
                                     kernel_initializer=tf.random_normal_initializer(stddev=post_scale),
                                     name="mlp1")
        x = inp + post_a_mlp
        if post_sa_layer_norm:
            with tf.variable_scope('post_a_layernorm'):
                x = tf.contrib.layers.layer_norm(x, begin_norm_axis=3)
        if n_mlp > 1:
            mlp = x
            mlp2_scale = np.sqrt(mlp_w1 / n_embd)
            mlp = tf.layers.dense(mlp,
                                  n_embd,
                                  kernel_initializer=tf.random_normal_initializer(stddev=mlp2_scale),
                                  name="mlp2")
        if n_mlp > 2:
            mlp3_scale = np.sqrt(mlp_w2 / n_embd)
            mlp = tf.layers.dense(mlp,
                                  n_embd,
                                  kernel_initializer=tf.random_normal_initializer(stddev=mlp3_scale),
                                  name="mlp3")
        if n_mlp > 1:
            x = x + mlp
        return x


def self_attention(inp, mask, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01,
                   scope='', reuse=False):
    '''
        Self attention over entities.
        Notation:
            T  - Time
            NE - Number entities
        Args:
            inp (tf) -- tensor w/ shape (bs, T, NE, features)
            mask (tf) -- binary tensor with shape (bs, T, NE). For each batch x time,
                            nner matrix represents entity i's ability to see entity j
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            qk_w, v_w (float) -- scale for gaussian init for keys/queries and values
                Std will be sqrt(scale/n_embd)
            scope (string) -- tf scope
            reuse (bool) -- tf reuse
    '''
    with tf.variable_scope(scope, reuse=reuse):
        bs, T, NE, features = shape_list(inp)
        # Put mask in format correct for logit matrix
        entity_mask = None
        if mask is not None:
            with tf.variable_scope('expand_mask'):
                assert np.all(np.array(mask.get_shape().as_list()) == np.array(inp.get_shape().as_list()[:3])),\
                    f"Mask and input should have the same first 3 dimensions. {shape_list(mask)} -- {shape_list(inp)}"
                entity_mask = mask
                mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)

        query, key, value = qkv_embed(inp, heads, n_embd, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w, reuse=reuse)
        logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
        logits /= np.sqrt(n_embd / heads)
        softmax = stable_masked_softmax(logits, mask)
        att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)
        with tf.variable_scope('flatten_heads'):
            out = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
            n_output_entities = shape_list(out)[2]
            out = tf.reshape(out, (bs, T, n_output_entities, n_embd))  # (bs, T, n_output_entities, n_embd)

        return out


def stable_masked_softmax(logits, mask):
    '''
        Args:
            logits (tf): tensor with shape (bs, T, heads, NE, NE)
            mask (tf): tensor with shape(bs, T, 1, NE)
    '''
    with tf.variable_scope('stable_softmax'):
        #  Subtract a big number from the masked logits so they don't interfere with computing the max value
        if mask is not None:
            mask = tf.expand_dims(mask, 2)
            logits -= (1.0 - mask) * 1e10

        #  Subtract the max logit from everything so we don't overflow
        logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
        unnormalized_p = tf.exp(logits)

        #  Mask the unnormalized probibilities and then normalize and remask
        if mask is not None:
            unnormalized_p *= mask
        normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
        if mask is not None:
            normalized_p *= mask
    return normalized_p


def qkv_embed(inp, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01, reuse=False):
    '''
        Compute queries, keys, and values
        Args:
            inp (tf) -- tensor w/ shape (bs, T, NE, features)
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            qk_w (float) -- Initialization scale for keys and queries. Actual scale will be
                sqrt(qk_w / #input features)
            v_w (float) -- Initialization scale for values. Actual scale will be sqrt(v_w / #input features)
            reuse (bool) -- tf reuse
    '''
    with tf.variable_scope('qkv_embed'):
        bs, T, NE, features = shape_list(inp)
        if layer_norm:
            with tf.variable_scope('pre_sa_layer_norm'):
                inp = tf.contrib.layers.layer_norm(inp, begin_norm_axis=3)

        # qk shape (bs x T x NE x h x n_embd/h)
        qk_scale = np.sqrt(qk_w / features)
        qk = tf.layers.dense(inp,
                             n_embd * 2,
                             kernel_initializer=tf.random_normal_initializer(stddev=qk_scale),
                             reuse=reuse,
                             name="qk_embed")  # bs x T x n_embd*2
        qk = tf.reshape(qk, (bs, T, NE, heads, n_embd // heads, 2))

        # (bs, T, NE, heads, features)
        query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]

        v_scale = np.sqrt(v_w / features)
        value = tf.layers.dense(inp,
                                n_embd,
                                kernel_initializer=tf.random_normal_initializer(stddev=v_scale),
                                reuse=reuse,
                                name="v_embed")  # bs x T x n_embd
        value = tf.reshape(value, (bs, T, NE, heads, n_embd // heads))

        query = tf.transpose(query, (0, 1, 3, 2, 4),
                             name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
        key = tf.transpose(key, (0, 1, 3, 4, 2),
                           name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
        value = tf.transpose(value, (0, 1, 3, 2, 4),
                             name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)

    return query, key, value


##################
# 1D Convolution #
##################

def circ_conv1d(inp, **conv_kwargs):
    valid_activations = {'relu': tf.nn.relu, 'tanh': tf.tanh, '': None}
    assert 'kernel_size' in conv_kwargs, f"Kernel size needs to be specified for circular convolution layer."
    conv_kwargs['activation'] = valid_activations[conv_kwargs['activation']]

    # concatenate input for circular convolution
    kernel_size = conv_kwargs['kernel_size']
    num_pad = kernel_size // 2
    inp_shape = shape_list(inp)
    inp_rs = tf.reshape(inp, shape=[inp_shape[0] * inp_shape[1]] + inp_shape[2:]) #  (BS * T, NE, feats)
    inp_padded = tf.concat([inp_rs[..., -num_pad:, :], inp_rs, inp_rs[..., :num_pad, :]], -2)
    out = tf.layers.conv1d(inp_padded,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           padding='valid',
                           **conv_kwargs)

    out = tf.reshape(out, shape=inp_shape[:3] + [conv_kwargs['filters']])
    return out

##################
# Misc ###########
##################


def layernorm(x, scope, epsilon=1e-5, reuse=False):
    '''
        normalize state vector to be zero mean / unit variance + learned scale/shift
    '''
    with tf.variable_scope(scope, reuse=reuse):
        n_state = x.get_shape()[-1]
        gain = tf.get_variable('gain', [n_state], initializer=tf.constant_initializer(1))
        bias = tf.get_variable('bias', [n_state], initializer=tf.constant_initializer(0))
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * gain + bias
