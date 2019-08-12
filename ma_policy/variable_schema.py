import numpy as np
import tensorflow as tf

BATCH = "batch"
TIMESTEPS = "timesteps"


class VariableSchema(object):
    def __init__(self, shape, dtype):
        """Creates a schema for a variable used in policy.
        Allows for symbolic definition of shape. Shape can consist of integers, as well as
        strings BATCH and TIMESTEPS. This is taken advantage of in the optimizers, to
        create placeholders or variables that asynchronously prefetch the inputs.

        Parameters
        ----------
        shape: [int, np.int64, np.int32, or str]
            shape of the variable, e.g. [12, 4], [BATCH, 12], [BATCH, 'timestep']
        dtype:
            tensorflow type of the variable, e.g. tf.float32, tf.int32
        """
        assert all(isinstance(s, (int, np.int64, np.int32)) or s in [BATCH, TIMESTEPS] for s in shape), 'Bad shape %s' % shape
        self.shape = shape
        self.dtype = tf.as_dtype(dtype)

    def _substituted_shape(self, batch=None, timesteps=None):
        feeds = dict(batch=batch, timesteps=timesteps)
        return [feeds.get(v, v) for v in self.shape]

    def substitute(self, *, batch=BATCH, timesteps=TIMESTEPS):
        """Make a new VariableSchema with batch or timesteps optionally filled in."""
        # Coerse None to default value.
        batch = batch or BATCH
        timesteps = timesteps or TIMESTEPS
        shape = self._substituted_shape(batch, timesteps)
        return VariableSchema(shape=shape, dtype=self.dtype)

    def placeholder(self, *, batch=None, timesteps=None, name=None):
        real_shape = self._substituted_shape(batch, timesteps)
        return tf.placeholder(self.dtype, real_shape, name=name)

    def variable(self, *, name, batch=None, timesteps=None, **kwargs):
        real_shape = self._substituted_shape(batch, timesteps)
        assert None not in real_shape
        return tf.get_variable(name, real_shape, self.dtype, **kwargs)

    def np_zeros(self, *, batch=None, timesteps=None, **kwargs):
        real_shape = self._substituted_shape(batch, timesteps)
        np_dtype = self.dtype.as_numpy_dtype
        return np.zeros(shape=real_shape, dtype=np_dtype, **kwargs)

    def match_shape(self, shape, *, batch=None, timesteps=None):
        expected = self._substituted_shape(batch, timesteps)
        if len(expected) != len(shape):
            return False
        for expected, actual in zip(expected, shape):
            if expected is not None and expected != actual:
                return False
        return True
