from tensorflow.python.framework import dtypes
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseMonotonicAttentionMechanism, _monotonic_probability_fn
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, nn_ops, variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
import numpy as np
import functools
import math
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score
coef = tf.constant(np.sqrt(1 / (2 * np.pi)), dtype=tf.float32)


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer, mu_prev):
    """Computes the attention and alignments for a given attention_mechanism."""

    alignments, next_attention_state, mu = attention_mechanism(cell_output, state=attention_state, mu_prev=mu_prev)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context_ = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context_ = array_ops.squeeze(context_, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context_], 1))
    else:
        attention = context_
    """
        attention is the context vector
        alignments is the weights of encoder hidden states
        next_attention_state is the weight of encoder hidden states
    """
    return attention, alignments, next_attention_state, mu


class GMMAttention(_BaseMonotonicAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               custom_key_value_fn=None,
               name="BahdanauAttention"):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is `tf.nn.softmax`. Other options include
        `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      custom_key_value_fn: (optional): The custom function for
        computing keys and values.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    # wrapped_probability_fn = lambda score, _: probability_fn(score)
    wrapped_probability_fn = functools.partial(_monotonic_probability_fn, sigmoid_noise=0, mode="parallel", seed=None)
    super(GMMAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        custom_key_value_fn=custom_key_value_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self.k = 5
    self.eps = 1e-5
    j = tf.expand_dims(tf.range(0, tf.shape(self.keys)[1]), 0)
    j = tf.tile(tf.expand_dims(j, axis=2), [self.batch_size, 1, self.k])
    self.j = tf.cast(j, dtype=tf.float32)


  def __call__(self, query, state, mu_prev):
    """Score the query based on the keys and values.

    Args:
      query: Tensor of dtype matching `self.values` and shape `[batch_size,
        query_depth]`.
      state: Tensor of dtype matching `self.values` and shape `[batch_size,
        alignments_size]` (`alignments_size` is memory's `max_time`).
      mu_prev: Tensor of dtype matching 'self.values' and shape '[batch_size, alignments_size, k]'
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "GMM_attention", [query]):

        # processed_query = self.query_layer(query) if self.query_layer else query
        # processed_query: [batch_size, 128]
        query_layer = layers_core.Dense(self.k*3,
                                        name="query_layer", use_bias=True, dtype=tf.float32)
        # hat_ods: [batch_size, kx3]
        hat_ods = query_layer(query)
        # hat_omega.shape == hat_sigma.shape == hat_delta.shape == [batch_size, k]
        hat_omega, hat_sigma, hat_delta = tf.split(hat_ods, num_or_size_splits=3, axis=1)

        # sigma.shape == [batch_size, k]
        sigma = tf.math.softplus(hat_sigma) + self.eps

        # delta.shape = [batch_size, k]
        delta = tf.math.softplus(hat_delta)

        # omega.shape = [batch_size, k]
        omega = tf.math.softmax(hat_omega, dim=-1)
        # coef = tf.constant(np.sqrt(1 / (2 * np.pi)), dtype=tf.float32)

        # omega_divide_z.shape = [batch_size, k]
        omega_divide_z = coef * (omega/sigma + self.eps)

        # expand_sigma.shape = [batch_size, alignments_size, k]
        expand_sigma = tf.tile(tf.expand_dims(sigma, 1), [1, self.alignments_size, 1])

        # j.shape = [batch_size, alignment_size, k]
        # j = tf.expand_dims(tf.range(0, tf.shape(self.keys)[1]), 0)
        # j = tf.tile(tf.expand_dims(j, axis=2), [self.batch_size, 1, self.k])
        # j = tf.cast(j, dtype=tf.float32)

        mu = mu_prev + tf.tile(tf.expand_dims(delta, 1), [1, self.alignments_size, 1])
        # mu.shape = [batch_size, alignment_size, k]

        expand_omega_divide_z = tf.tile(tf.expand_dims(omega_divide_z, 1), [1, self.alignments_size, 1])
        tmp = tf.math.exp(-0.5 * (self.j-mu) ** 2/expand_sigma ** 2)

        # alignments = tf.math.softmax(tf.reduce_sum(expand_omega_divide_z * tmp, axis=-1), axis=-1)
        # alignments = self._probability_fn(tf.reduce_sum(expand_omega_divide_z * tmp, axis=-1), state)
        alignments = tf.reduce_sum(expand_omega_divide_z * tmp, axis=-1)
        next_state = alignments

        return alignments, next_state, mu
