import tensorflow as tf


def _mean_std_update_size(x, axes):
    x_shape = tf.shape(x)
    x_dims_to_reduce = tf.gather(x_shape, axes)
    size = tf.reduce_prod(x_dims_to_reduce)
    return size


def _interpolate(old, new, old_weight, scaled_weight):
    return old * old_weight + new * scaled_weight


def _std_from_mean_and_square(mean, square):
    var_est = tf.to_float(square) - tf.square(mean)
    return tf.sqrt(tf.maximum(var_est, 1e-2))


class EMAMeanStd(object):
    """
    Calculates an Exponential Moving Average for each argument with
    exponential coefficient `beta`. The forward relation is:
        mean = beta * old_mean + (1.0 - beta) * observation
    The algorithm removes the bias introduced from setting ema[-1] = 0.0

    Note: `beta` parameter is defined with respect to a single observation within a batch
    if `per_element_update=True` (if a batch has 1000 elements of an observation, this is
    considered to be a 1000 updates), else it is considered to be the size of an update for a full
    batch (1 update if `per_element_update=False`).
    """

    def __init__(self, beta, scope="ema", reuse=None, epsilon=1e-6, per_element_update=False, shape=(), version=1):
        self._version = version
        self._per_element_update = per_element_update
        with tf.variable_scope(scope, reuse=reuse):
            # Expected value of x
            self._biased_mean = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="mean",
                trainable=False)
            # Expected value of x^2
            self._biased_sq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sq",
                trainable=False)
            # How to integrate observations of x over time
            self._one_minus_beta = 1.0 - beta
            # Weight placed on ema[-1] == 0.0 which we divide out to debias
            self._debiasing_term = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="debiasing_term",
                trainable=False)
            self.shape = shape

            # the stored mean and square are biased due to setting ema[-1] = 0.0,
            # we correct for this by dividing by the debiasing term:
            self.mean = self._biased_mean / tf.maximum(self._debiasing_term, epsilon)
            self.std = _std_from_mean_and_square(mean=self.mean, square=self._biased_sq / tf.maximum(self._debiasing_term, epsilon))

    def update_op(self, x, axes=(0,)):
        scaled_weight = tf.cast(self._one_minus_beta, tf.float64)
        if self._per_element_update:
            # many updates were done at once in a batch, so we figure out what power
            # to raise `1-beta` to.
            # using the fact that for small 1.0 - beta we have:
            # 1 - beta^N ~= (1.0 - beta) * N
            size = _mean_std_update_size(x, axes)
            scaled_weight *= tf.cast(size, tf.float64)
        one = tf.constant(1.0, dtype=tf.float64)
        old_weight = one - scaled_weight
        old_weight_fp32 = tf.to_float(old_weight)
        scaled_weight_fp32 = tf.to_float(scaled_weight)
        return tf.group(
            # increment the running debiasing term by the contribution of the initial ema[-1] == 0.0 observation
            # (e.g. boost the observed value by how much it was initially discounted on step 1)
            tf.assign(self._debiasing_term, tf.to_float(_interpolate(old=tf.cast(self._debiasing_term, tf.float64), new=one, old_weight=old_weight, scaled_weight=scaled_weight))),
            # do an interpolation on the expected value of X
            tf.assign(self._biased_mean, _interpolate(old=self._biased_mean, new=tf.reduce_mean(tf.to_float(x), axis=axes), old_weight=old_weight_fp32, scaled_weight=scaled_weight_fp32)),
            # do an interpolation on the expected value of X^2
            tf.assign(self._biased_sq, _interpolate(old=self._biased_sq, new=tf.reduce_mean(tf.square(tf.to_float(x)), axis=axes), old_weight=old_weight_fp32, scaled_weight=scaled_weight_fp32)),
        )
