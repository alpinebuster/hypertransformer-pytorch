"""Transformer model."""

import dataclasses

from typing import Optional, Tuple, Callable

import tensorflow.compat.v1 as tf # pyright: ignore[reportMissingImports] # pylint:disable=import-error
import torch
import torch.nn as nn
import torch.nn.functional as F

LARGE_NUMBER = 1e8

TwoTensors = Tuple[tf.Tensor, tf.Tensor]
ThreeTensors = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
ActivationFn = Callable[[tf.Tensor], tf.Tensor]


@dataclasses.dataclass
class TransformerParams:
    """Transformer model parameters.

    Attributes:
       query_key_dim (D_qk): the dimension of the query/key
       value_dim (D_v): the dimension of the value embedding (defaults to query_key_dim)
       internal_dim: dimension of the embedding in pointwise layer
       num_layers: number of transformer layers
       mha_output_dim (D_mha_out): the dimension of the final output of pointwise layer (Multi-Head Attention)
       heads (H): number of heads in multi-head attention value_dim and query_key_dim
         need to be divisible by
       dropout_rate: dropout applied to the output of each transformer block
       activation_fn: activation to use in feed forward blocks.
       attention_activation_fn: activation function to use in the attention
         module (default is softmax).
       
       e.g. Multi-Head Attention (MHA)
       1. Input (T_q/T_k/T_v: the sequence length of query/key/value, and for self-attention: `T_q = T_k = T_v = T`)
        X_q ∈ ℝ[B, T_q, D_model] (D_model is the embedding / hidden dimension of the model)
        X_k ∈ ℝ[B, T_k, D_model]
        X_v ∈ ℝ[B, T_v, D_model]
            ↓
       2. Linear projection
        Q = X_q × W_Q → [B, T_q, D_qk], W_Q ∈ ℝ[D_model × D_qk]
        K = X_k × W_K → [B, T_k, D_qk], W_K ∈ ℝ[D_model × D_qk]
        V = X_v × W_V → [B, T_v, D_v],  W_V ∈ ℝ[D_model × D_v]
            ↓
       3. Split heads
        Q = [B, T_q, H*d_qk] → reshape → [B, T_q, H, d_qk] → transpose → [B, H, T_q, d_qk]
        K = [B, T_q, H*d_qk] → reshape → [B, T_k, H, d_qk] → transpose → [B, H, T_k, d_qk]
        V = [B, T_v, H*d_v]  → reshape → [B, T_v, H, d_v]  → transpose → [B, H, T_v, d_v]
            ↓
       4. Attention scores (QK^T)
        scores = [B, H, T_q, d_qk] × [B, H, d_qk, T_k] → [B, H, T_q, T_k] → scores /= √d_qk
            ↓
       5. Attention Activation (Softmax) 
        α = Softmax(scores, axis=T_k) → [B, H, T_q, T_k]
            ↓
       6. Weighted sum (·V)
        context = α × V → [B, H, T_q, T_k] × [B, H, T_k, d_v] → [B, H, T_q, d_v]
            ↓
       7. Transpose
        context = [B, H, T_q, d_v] → [B, T_q, H, d_v]
            ↓
       8. Concat multiple heads
        [B, T_q, H, d_v] → reshape → [B, T_q, H*d_v] = [B, T_q, D_v]
            ↓
       9. Output projection
        [B, T_q, D_v] → [B, T_q, D_mha_out]
    """

    query_key_dim: int
    internal_dim: int
    num_layers: int
    value_dim: Optional[int] = None
    mha_output_dim: Optional[int] = None
    heads: int = 1
    dropout_rate: float = 0.1
    activation_fn: ActivationFn = tf.nn.relu
    attention_activation_fn: ActivationFn = tf.nn.softmax

    def __post_init__(self):
        if self.value_dim is None:
            self.value_dim = self.query_key_dim
        if self.mha_output_dim is None:
            self.mha_output_dim = self.value_dim
        assert self.value_dim % self.heads == 0
        assert self.query_key_dim % self.heads == 0


def attention(q, k, v, mask=None, act_fn=tf.nn.softmax):
    """Simple attention module.

    Args:
      q: batch × head × seq × d_qk
      k: batch × head × seq × d_qk
      v: batch × head × seq × d_v
      mask: which examples in seq to ignore
      act_fn: Activation function to use for attention (defaults to softmax).
    Returns:
      batch × head × seq × d_v
    """
    # [B, H, S, d_qk] × [B, H, d_qk, S] → [B, H, S, S]
    attention_product = tf.matmul(q, k, transpose_b=True)
    key_dim = tf.cast(tf.shape(k)[-1], tf.float32)
    attention_logits = attention_product / tf.math.sqrt(key_dim)

    if mask is not None:
        # attention_logits[masked_positions] = -∞ → softmax(-∞) = 0
        attention_logits -= mask * LARGE_NUMBER

    attention_weights = act_fn(attention_logits)
    # [B, H, S, S] × [B, H, S, d_v] → [B, H, S, d_v]
    return tf.matmul(attention_weights, v), attention_weights


class PWFeedForward(tf.Module):
    """Pointwise feedforward layer.

    In each Transformer layer, the structure is typically:

       Attention → [Dropout + Add & Norm] → (PW)FeedForward → [Dropout + Add & Norm]

        input_tensor
          ↓
        Dense(layer_1): input_dim → internal_dim + activation
          ↓
        Dense(layer_2): internal_dim → mha_output_dim (dim)
          ↓
        output

    Key Characteristic:
       The same MLP is applied independently to each token (patch / position). It is used after the attention module to introduce non-linearity and enhance channel-wise representational capacity, while not introducing any interaction between different tokens.

    > For this reason, it is referred to as point-wise (or position-wise) feed-forward.
    """

    def __init__(
        self,
        dim: Optional[int], # mha_output_dim
        internal_dim: int,
        name: Optional[str]=None,
        activation=tf.nn.relu,
    ):
        super().__init__(name=name)

        # [..., input_dim] -> [..., internal_dim]
        self.layer_1 = tf.layers.Dense(
            internal_dim, activation=activation, name="layer_1"
        )
        # [..., internal_dim] -> [..., dim]
        self.layer_2 = tf.layers.Dense(dim, name="layer_2")

    def __call__(self, input_tensor):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            return self.layer_2(self.layer_1(input_tensor))


class MultiHeadAttention(tf.Module):
    """Multi-head attention layer."""

    def __init__(self, params: TransformerParams, name: Optional[str] = None):
        super().__init__(name=name)

        self.num_heads = params.heads
        self.qk_depth = params.query_key_dim // params.heads
        assert params.value_dim
        self.v_depth = params.value_dim // params.heads
        self.v_dim = params.value_dim

        self.w_q = tf.layers.Dense(params.query_key_dim, name="q")
        self.w_k = tf.layers.Dense(params.query_key_dim, name="k")
        self.w_v = tf.layers.Dense(params.value_dim, name="v")
        self.dense = tf.layers.Dense(params.mha_output_dim, name="fc")
        self.attn_act_fn = params.attention_activation_fn

    def _split_heads(self, x, batch_size: int, depth: int):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask=None):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            batch_size = tf.shape(q)[0]
            # Q = X_q × W_Q → [B, T_q, D_qk], W_Q ∈ ℝ[D_model × D_qk]
            # K = X_k × W_K → [B, T_k, D_qk], W_K ∈ ℝ[D_model × D_qk]
            # V = X_v × W_V → [B, T_v, D_v],  W_V ∈ ℝ[D_model × D_v]
            q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
            # [B, T_q, H, d_qk]
            q = self._split_heads(q, batch_size, self.qk_depth)
            # [B, T_q, H, d_qk]
            k = self._split_heads(k, batch_size, self.qk_depth)
            # [B, T_v, H, d_v]
            v = self._split_heads(v, batch_size, self.v_depth)
            # [B, H, T_q, d_v]
            scaled_attention, attention_weights = attention(
                q, k, v, mask, act_fn=self.attn_act_fn
            )
            # [B, T_q, H, d_v]
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            # [B, T_q, H, d_v] → reshape → [B, T_q, H*d_v] = [B, T_q, D_v]
            concat_attention = tf.reshape(
                scaled_attention, (batch_size, -1, self.v_dim)
            )
            # [B, T_q, D_v] → [B, T_q, D_mha_out]
            return self.dense(concat_attention), attention_weights


class EncoderLayer(tf.Module):
    """Encoder layer.

    The structure of a standard Transformer Encoder Layer is:
       Input x
        ↓
       [Multi-Head Attention]
        ↓
       [Dropout + Add & Norm]     ← Residual Connection + LayerNorm
        ↓
       [PWFeedForward]            ← FFN: Dense → Activation → Dense
        ↓
       [Dropout + Add & Norm]     ← Residual Connection + LayerNorm
        ↓
       Output
    """

    def __init__(self, params: TransformerParams, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)

        self.mha = MultiHeadAttention(params, name="attention")
        self.ffn = None
        if params.internal_dim > 0:
            self.ffn = PWFeedForward(
                dim=params.mha_output_dim,
                internal_dim=params.internal_dim,
                activation=params.activation_fn,
                name="fc",
            )

        """
        1. BatchNorm
           batch
             ↓
          [x1_d1  x1_d2  x1_d3]
          [x2_d1  x2_d2  x2_d3]  → Calculate the average values of d1, d2 and d3 respectively
          [x3_d1  x3_d2  x3_d3]

        2. LayerNorm
          [x1_d1  x1_d2  x1_d3]  → Calculate the average of all the features of x1
          [x2_d1  x2_d2  x2_d3]  → Calculate the average of all the features of x2
          [x3_d1  x3_d2  x3_d3]
        """
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.layers.Dropout(params.dropout_rate)
        self.dropout_2 = tf.layers.Dropout(params.dropout_rate)

    def __call__(self, x, is_training=True, mask=None):
        # Use `name_scope` to manage operation names (making the computation graph clearer and easier to visualize)
        # Use `variable_scope` to manage variable names and reuse (ensuring parameters are not created repeatedly)
        # 
        # tf.variable_scope(name_or_scope=None, default_name=self.name)
        #       ↓
        #   current_scope/
        #     + default_name/
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            attn_output, attention_weights = self.mha(x, x, x, mask)
            self.attention_weights = attention_weights
            attn_output = self.dropout_1(attn_output, training=is_training)
            out_1 = self.layer_norm_1(x + attn_output)
            if self.ffn is None:
                return out_1
            ffn_output = self.ffn(out_1)
            ffn_output = self.dropout_2(ffn_output, training=is_training)
            return self.layer_norm_2(out_1 + ffn_output)


class DecoderLayer(tf.Module):
    """Decoder layer.

    The structure of a standard Transformer Decoder Layer is:
       Input x
        ↓
       Masked Multi-Head Self-Attention
        ↓
       Dropout
        ↓
       Add & LayerNorm
        ↓
       Encoder–Decoder Attention (Cross Attention)
        ↓
       Dropout
        ↓
       Add & LayerNorm
        ↓
       Feed Forward Network (FFN: Dense → Act → Dense)
        ↓
       Dropout
        ↓
       Add & LayerNorm
        ↓
       Output

    > It is similar to the `EncoderLayer`, but it has an additional key module: `Encoder-Decoder Attention` (Cross Attention) and padding mask + look-ahead (causal) mask to prevent attending to future positions (Encoder only has padding mask).
    """

    def __init__(self, params: TransformerParams, name: Optional[str] = None):
        super().__init__(name=name)

        self.mha_1 = MultiHeadAttention(params, name="attention_1")
        self.mha_2 = MultiHeadAttention(params, name="attention_2")
        self.ffn = None
        if params.internal_dim > 0:
            self.ffn = PWFeedForward(
                dim=params.mha_output_dim,
                internal_dim=params.internal_dim,
                name="fc",
            )

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.layers.Dropout(params.dropout_rate)
        self.dropout_2 = tf.layers.Dropout(params.dropout_rate)
        self.dropout_3 = tf.layers.Dropout(params.dropout_rate)

    def __call__(
        self, x, enc_output, is_training=True, look_ahead_mask=None, padding_mask=None
    ):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            # Masked Self-Attention: `look_ahead_mask` -> Mask future tokens
            attn_1, attn_weights_block_1 = self.mha_1(x, x, x, look_ahead_mask)
            attn_1 = self.dropout_1(attn_1, training=is_training)
            # Residual Connection + LayerNorm
            out_1 = self.layer_norm_1(attn_1 + x)

            # Encoder–Decoder Attention (Cross Attention)
            # The Decoder "queries" the output of the Encoder based on the current generation status
            attn_2, attn_weights_block_2 = self.mha_2(
                enc_output, # v
                enc_output, # k
                out_1,      # q
                padding_mask, # padding_mask -> Prevent focusing on padding in the Encoder `enc_output`
            )
            attn_2 = self.dropout_2(attn_2, training=is_training)
            out_2 = self.layer_norm_2(attn_2 + out_1)

            if self.ffn is not None:
                ffn_output = self.ffn(out_2)
                ffn_output = self.dropout_3(ffn_output, training=is_training)
                out_3 = self.layer_norm_3(ffn_output + out_2)
            else:
                out_3 = out_2

        return out_3, attn_weights_block_1, attn_weights_block_2


class Encoder(tf.Module):
    """Transformer encoder.

    The structure of a standard Transformer Encoder is:
       Input x
        ↓
       EncoderLayer 1
        ↓
       EncoderLayer 2
        ↓
       ...
        ↓
       EncoderLayer N (optionally without PW-FFN nonlinearity, if `skip_last_nonlinearity=True`)
        ↓
       Output `enc_output`

    Notes:
    - When `skip_last_nonlinearity` is True, the Feed-Forward Network (FFN)
    in the last EncoderLayer uses no activation function (i.e., a linear FFN).
    - This is sometimes used when a more linear output space is desired.
    """

    def __init__(
        self,
        params: TransformerParams,
        layer_dropout_prob=0.0,
        skip_last_nonlinearity=False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.num_layers = params.num_layers
        self.enc_layers = [
            EncoderLayer(params, name=f"layer_{i+1}")
            for i in range(params.num_layers - 1)
        ]

        if skip_last_nonlinearity:
            params = dataclasses.replace(
                params, activation_fn=None
            )
        self.enc_layers.append(EncoderLayer(params, name=f"layer_{params.num_layers}"))

        self.layer_dropout_prob = layer_dropout_prob

    def __call__(self, x, is_training: bool, mask=None):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            # Layer dropout only takes effect during the training stage
            dropout_prob = self.layer_dropout_prob if is_training else 0
            for i in range(self.num_layers):
                # `tf.random_uniform((), 0, 1)` -> 0.13, 0.87, 0.42, ...
                # `tf.cast(..., tf.float32)`    -> True 👉 1, False 👉 0
                # `select` ∈ {0.0, 1.0}
                select = tf.cast(tf.random_uniform((), 0, 1) > dropout_prob, tf.float32)
                # If select = 1 -> x = EncoderLayer_i(x), 
                # if select = 0 -> x = x (Residual Connection), 👉 the entire layer is randomly discarded
                x = self.enc_layers[i](x, is_training, mask) * select + x * (1 - select)
        return x


class Decoder(tf.Module):
    """Transformer decoder.

    The structure of a standard Transformer Encoder is:
       Input x
        ↓
       DecoderLayer 1
        ↓
       DecoderLayer 2
        ↓
       ...
        ↓
       DecoderLayer N (optionally without PW-FFN nonlinearity, if `skip_last_nonlinearity=True`)
        ↓
       Output `enc_output`

    Differences from the Encoder:
       - Each DecoderLayer contains TWO attention sublayers:
          1) Masked self-attention over the inputs.
          2) Encoder–decoder cross attention over the encoder outputs.
          3) DecoderLayers include an additional attention sublayer (encoder–decoder attention),
             and therefore one additional Dropout and LayerNorm compared to EncoderLayers.
       - The self-attention in the decoder use padding mask + (causal) look-ahead mask (to prevent attending to future positions), while the encoder self-attention only applies padding masks.
       - The decoder attends to the encoder output (`enc_output`) via cross-attention, whereas the encoder only performs self-attention.
    """

    def __init__(
        self,
        params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        self.num_layers = params.num_layers
        self.dec_layers = [
            DecoderLayer(params, name=f"layer_{i+1}")
            for i in range(params.num_layers - 1)
        ]
        if skip_last_nonlinearity:
            params = dataclasses.replace(
                params, activation_fn=None
            )
        self.dec_layers.append(DecoderLayer(params, name=f"layer_{params.num_layers}"))

    def __call__(
        self, x, enc_output, is_training: bool, look_ahead_mask=None, padding_mask=None
    ):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            for i in range(self.num_layers):
                x, _, _ = self.dec_layers[i](
                    x, enc_output, is_training, look_ahead_mask, padding_mask
                )
        return x


class EncoderDecoderModel(tf.Module):
    """Transformer model."""

    def __init__(
        self,
        params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        self.encoder = Encoder(params, name="encoder")
        self.decoder = Decoder(
            params,
            name="decoder",
            skip_last_nonlinearity=skip_last_nonlinearity,
        )

    def __call__(self, sequence, mask=None, is_training=True):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            # [batch_size, seq_len, hidden_dim]    ← Normal conditions
            # [seq_len, hidden_dim]                ← NO batch dimension
            single_sequence = len(sequence.shape) == 2
            if single_sequence:
                sequence = tf.expand_dims(sequence, axis=0)

            encoding = self.encoder(sequence, is_training=is_training, mask=mask)
            output = self.decoder(
                sequence,
                encoding,
                is_training=is_training,
                look_ahead_mask=mask,
                padding_mask=mask,
            )
            if single_sequence:
                output = tf.squeeze(output, axis=0)
        return output


class EncoderModel(tf.Module):
    """Transformer model."""

    def __init__(
        self,
        params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.encoder = Encoder(
            params, name="encoder", skip_last_nonlinearity=skip_last_nonlinearity
        )

    def __call__(self, sequence, mask=None, is_training=True):
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            # [batch_size, seq_len, hidden_dim]    ← Normal conditions
            # [seq_len, hidden_dim]                ← NO batch dimension
            single_sequence = len(sequence.shape) == 2
            if single_sequence:
                sequence = tf.expand_dims(sequence, axis=0)

            output = self.encoder(sequence, mask=mask, is_training=is_training)
            if single_sequence:
                output = tf.squeeze(output, axis=0)
        return output


class SeparateEncoderDecoderModel(tf.Module):
    """Model using encoder for samples and decoder for weights."""

    def __init__(
        self,
        encoder_params: TransformerParams,
        decoder_params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.encoder = Encoder(encoder_params, name="encoder")
        self.decoder = Decoder(
            decoder_params,
            name="decoder",
            skip_last_nonlinearity=skip_last_nonlinearity,
        )

    def __call__(self, sample_sequence, weight_sequence, mask=None, is_training=True):
        assert mask is None
        with self.name_scope, tf.variable_scope(name_or_scope=None, default_name=self.name):
            # [batch_size, seq_len, hidden_dim]    ← Normal conditions
            # [seq_len, hidden_dim]                ← NO batch dimension
            single_sequence = len(sample_sequence.shape) == 2
            if single_sequence:
                sample_sequence = tf.expand_dims(sample_sequence, axis=0)
                weight_sequence = tf.expand_dims(weight_sequence, axis=0)

            encoding = self.encoder(sample_sequence, is_training=is_training)
            output = self.decoder(weight_sequence, encoding, is_training=is_training)
            if single_sequence:
                output = tf.squeeze(output, axis=0)
        return output
