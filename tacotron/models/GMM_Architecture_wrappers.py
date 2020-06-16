import collections

import numpy as np
import tensorflow as tf
# from tacotron.models.attention import _compute_attention
from tacotron.models.simple_bahdanau_attention import _compute_attention
# from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.util import nest

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class TacotronGMMDecoderCellState(collections.namedtuple("TacotronGMMDecoderCellState",
                                                         ("cell_state", "attention", "time", "alignments",
                                                          "alignment_history", "mu"))):

    def replace(self, **kwargs):
        """Clones the current state while overwriting components provided by kwargs.
        """
        return super(TacotronGMMDecoderCellState, self)._replace(**kwargs)


class TacotronGMMDecoderCell(RNNCell):
    """
        Tactron 2 Decoder Cell
        Decodes encoder output and previous mel frames into next r frames
        Decoder step i:
        1) Prenet to compress last output information
        2) Concat compressed inputs with previous context vector (input feeding) *
        3) Decoder RNN (actual decoding) to predict current state s_{i} *
        4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
        5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)
        6) Predict <stop_token> output ys_{i} using s_{i} and c_{i} (concatenated)
        * : This is typically taking a vanilla LSTM, wrapping it using tensorflow's attention wrapper,
        and wrap that with the prenet before doing an input feeding, and with the prediction layer
        that uses RNN states to project on output space. Actions marked with (*) can be replaced with
        tensorflow's attention wrapper call if it was using cumulative alignments instead of previous alignments only.
    """

    def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection, k=5):
        super(TacotronGMMDecoderCell, self).__init__()
        self._prenet = prenet
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection
        self.k = k
        self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size,
                                       self._attention_mechanism.batch_size,
                                       message=error_message)]

    @property
    def output_size(self):
        return self._frame_projection.shape

    @property
    def state_size(self):
        """The `state_size` property of `TacotronDecoderCell`.

        Returns:
                An `TacotronGMMDecoderCellState` tuple containing shapes used by this object.
        """

        return TacotronGMMDecoderCellState(cell_state=self._cell._cell.state_size,
                                           attention=self._attention_layer_size,
                                           time=tensor_shape.TensorShape([]),
                                           alignments=self._attention_mechanism.alignments_size,
                                           alignment_history=(),
                                           mu=(array_ops.shape(self._attention_mechanism.keys)[0],
                                               array_ops.shape(self._attention_mechanism.keys)[1], self.k))

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

            Args:
                batch_size: `0D` integer tensor: the batch size.
                dtype: The internal state data type.
                Returns:
                    An `TacotronGMMDecoderCellState` tuple containing zeroed out tensors and,
                    possibly, empty `TensorArray` objects.
                Raises:
                    ValueError: (or, possibly at runtime, InvalidArgument), if
                    `batch_size` does not match the output size of the encoder passed
                    to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self._cell._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of TacotronGMMDecoderCellState %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.")

            mu_shape = (array_ops.shape(self._attention_mechanism.keys)[0],
                        array_ops.shape(self._attention_mechanism.keys)[1], self.k)
            with ops.control_dependencies(
                self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            return TacotronGMMDecoderCellState(cell_state=cell_state,
                                               attention=_zero_state_tensors(self._attention_layer_size,
                                                                             batch_size, dtype),
                                               time=array_ops.zeros([], dtype=tf.int32),
                                               alignments=self._attention_mechanism.initial_alignments(batch_size,
                                                                                                       dtype),
                                               alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
                                                                                              dynamic_size=True),
                                               mu=array_ops.zeros(shape=mu_shape, dtype=tf.float32))

    def __call__(self, inputs, state):
        """
        alignment is the weight of attention mechanism
        :param inputs:
        :param state:
        :return:
        """
        # Information bottleneck (essential for learning attention)
        prenet_output = self._prenet(inputs)

        # Concat context vector and prenet output to form LSTM cells input (input feeding)
        LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

        # Unidirectional LSTM layers
        LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)

        # Compute the attention (context) vector and alignments using
        # the new decoder cell hidden state as query vector
        # and cumulative alignments to extract location features
        # The choice of the new cell hidden state (s_{i}) of the last
        # decoder RNN Cell is based on Luong et Al. (2015):
        # https://arxiv.org/pdf/1508.04025.pdf

        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        mu_prev = state.mu
        context_vector, alignments, cumulated_alignments, mu = \
            _compute_attention(attention_mechanism=self._attention_mechanism, cell_output=LSTM_output,
                               attention_state=previous_alignments, attention_layer=None, mu_prev=mu_prev)

        projections_input = tf.concat([LSTM_output, context_vector], axis=-1)
        cell_outputs = self._frame_projection(projections_input)
        stop_tokens = self._stop_projection(projections_input)

        alignment_history = previous_alignment_history.write(state.time, alignments)

        next_state = TacotronGMMDecoderCellState(time=state.time + 1,
                                                 cell_state=next_cell_state,
                                                 attention=context_vector,
                                                 alignments=cumulated_alignments,
                                                 alignment_history=alignment_history,
                                                 mu=mu)

        return (cell_outputs, stop_tokens), next_state