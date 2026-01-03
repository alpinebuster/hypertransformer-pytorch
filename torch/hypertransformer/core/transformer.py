"""Transformer model."""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

LARGE_NUMBER = 1e8

InputDim = Union[Tuple[int], Tuple[int, int]]
TwoTensors = Tuple[torch.Tensor, torch.Tensor]
ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ActivationFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
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
       1. Input (S_q/S_k/S_v: the sequence length of query/key/value, and for self-attention: `S_q = S_k = S_v = S`)
        X_q ∈ ℝ[B, S_q, D_model] (D_model is the embedding / hidden dimension of the model)
        X_k ∈ ℝ[B, S_k, D_model]
        X_v ∈ ℝ[B, S_v, D_model]
            ↓
       2. Linear projection
        Q = X_q × W_Q → [B, S_q, D_qk], W_Q ∈ ℝ[D_model × D_qk]
        K = X_k × W_K → [B, S_k, D_qk], W_K ∈ ℝ[D_model × D_qk]
        V = X_v × W_V → [B, S_v, D_v],  W_V ∈ ℝ[D_model × D_v]
            ↓
       3. Split heads
        Q = [B, S_q, H*d_qk] → reshape → [B, S_q, H, d_qk] → transpose → [B, H, S_q, d_qk]
        K = [B, S_q, H*d_qk] → reshape → [B, S_k, H, d_qk] → transpose → [B, H, S_k, d_qk]
        V = [B, S_v, H*d_v]  → reshape → [B, S_v, H, d_v]  → transpose → [B, H, S_v, d_v]
            ↓
       4. Attention scores (QK^T)
        scores = [B, H, S_q, d_qk] × [B, H, d_qk, S_k] → [B, H, S_q, S_k] → scores /= √d_qk
            ↓
       5. Attention Activation (Softmax) 
        α = Softmax(scores, axis=S_k) → [B, H, S_q, S_k]
            ↓
       6. Weighted sum (·V)
        context = α × V → [B, H, S_q, S_k] × [B, H, S_k, d_v] → [B, H, S_q, d_v]
            ↓
       7. Transpose
        context = [B, H, S_q, d_v] → [B, S_q, H, d_v]
            ↓
       8. Concat multiple heads
        [B, S_q, H, d_v] → reshape → [B, S_q, H*d_v] = [B, S_q, D_v]
            ↓
       9. Output projection
        [B, S_q, D_v] → [B, S_q, D_mha_out]
    """

    query_key_dim: int
    internal_dim: int
    num_layers: int
    value_dim: Optional[int] = None
    mha_output_dim: Optional[int] = None
    heads: int = 1
    dropout_rate: float = 0.1
    activation_fn: ActivationFn = F.relu
    attention_activation_fn: ActivationFn = lambda x: torch.softmax(x, dim=-1)

    def __post_init__(self):
        if self.value_dim is None:
            self.value_dim = self.query_key_dim
        if self.mha_output_dim is None:
            self.mha_output_dim = self.value_dim

        assert self.value_dim % self.heads == 0
        assert self.query_key_dim % self.heads == 0


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor]=None,
    act_fn: ActivationFn = lambda x: torch.softmax(x, dim=-1),
) -> TwoTensors:
    """Simple attention module.

    Args:
      q: [B, H, S_q, d_qk]
      k: [B, H, S_k, d_qk]
      v: [B, H, S_k, d_v]
      mask: which examples in seq to ignore, broadcastable to [B, H, S_q, S_k], 0 means masked
      act_fn: Activation function to use for attention (defaults to softmax).

    Returns:
      context: [B, H, S_q, d_v]
      attention_weights: [B, H, S_q, S_k]
    """
    # [B, H, S_q, d_qk] × [B, H, d_qk, S_k] → [B, H, S_q, S_k]
    attention_product = torch.matmul(q, k.transpose(-2, -1))
    # d_qk == q.size(-1) == k.size(-1) -> Whichever one is used is completely equivalent
    # q.size(-1) = the dimension actually used by matmul
    attention_logits = attention_product * (q.size(-1) ** -0.5)

    if mask is not None:
        # mask == 0 → masked
        # attention_logits[masked_positions] = -∞ → softmax(-∞) = 0
        attention_logits -= mask * LARGE_NUMBER

    attention_weights = act_fn(attention_logits)
    # [B, H, S_q, S_k] × [B, H, S_k, d_v] → [B, H, S_q, d_v], [B, H, S_q, S_k]
    return torch.matmul(attention_weights, v), attention_weights


class PWFeedForward(nn.Module):
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
        output_dim: Optional[int], # mha_output_dim
        internal_dim: int,
        name: Optional[str] = None,
        activation: ActivationFn = torch.relu,
    ):
        super().__init__()

        self.name = name
        assert output_dim is not None
        # Called after MHA, so the input_dim = mha_output_dim
        self.input_dim = output_dim
        # [..., input_dim] -> [..., internal_dim]
        self.layer_1 = nn.Linear(self.input_dim, internal_dim)
        # [..., internal_dim] -> [..., output_dim]
        self.layer_2 = nn.Linear(internal_dim, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        if self.activation is None:
            return self.layer_2(self.layer_1(x))
        return self.layer_2(self.activation(self.layer_1(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        params: TransformerParams,
        name: Optional[str] = None
    ):
        super().__init__()

        self.name = name
        self.num_heads = params.heads
        self.qk_depth = params.query_key_dim // params.heads
        assert params.value_dim
        self.v_depth = params.value_dim // params.heads
        self.v_dim = params.value_dim

        assert params.mha_output_dim is not None
        # D_model is the transformer hidden size（embedding dim）
        # D_model = D_mha_out
        self.w_q = nn.Linear(params.mha_output_dim, params.query_key_dim)
        self.w_k = nn.Linear(params.mha_output_dim, params.query_key_dim)
        self.w_v = nn.Linear(params.mha_output_dim, params.value_dim)

        self.dense = nn.Linear(params.value_dim, params.mha_output_dim)
        self.attn_act_fn = params.attention_activation_fn

    def _split_heads(self, x: torch.Tensor, batch_size: int, depth: int) -> torch.Tensor:
        """
        x: [B, S, D]
        return: [B, H, S, depth]
        """
        x = x.view(batch_size, -1, self.num_heads, depth)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        x_v: torch.Tensor,
        x_k: torch.Tensor,
        x_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TwoTensors:
        # x_q,x_k,x_v -> [batch_size, seq_len, hidden_dim]
        batch_size = x_q.size(dim=0)

        # Q = x_q × W_Q → [B, S_q, D_qk], W_Q ∈ ℝ[hidden_dim × D_qk]
        # K = x_k × W_K → [B, S_k, D_qk], W_K ∈ ℝ[hidden_dim × D_qk]
        # V = x_v × W_V → [B, S_v, D_v],  W_V ∈ ℝ[hidden_dim × D_v]
        q, k, v = self.w_q(x_q), self.w_k(x_k), self.w_v(x_v)

        # [B, H, S_q, d_qk]
        q = self._split_heads(q, batch_size, self.qk_depth)
        # [B, H, S_q, d_qk]
        k = self._split_heads(k, batch_size, self.qk_depth)
        # [B, H, S_v, d_v]
        v = self._split_heads(v, batch_size, self.v_depth)

        # [B, H, S_q, d_v], [B, H, S_q, S_k]
        scaled_attention, attention_weights = attention(
            q, k, v, mask, act_fn=self.attn_act_fn
        )
        # [B, S_q, H, d_v]
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        # [B, S_q, H, d_v] → reshape → [B, S_q, H*d_v] = [B, S_q, D_v]
        concat_attention = scaled_attention.contiguous().view(
            batch_size, -1, self.v_dim
        )

        # [B, S_q, D_v] → [B, S_q, D_mha_out], [B, H, S_q, S_k]
        return self.dense(concat_attention), attention_weights


class EncoderLayer(nn.Module):
    """Encoder layer.

    The structure of a standard Transformer Encoder Layer is:
       Input x
        ↓
       [Multi-Head Attention]
        ↓
       [Dropout + Add & Norm]     ← Residual Connection + LayerNorm
        ↓
       [PWFeedForward]            ← (Optional) FFN: Dense → Activation → Dense
        ↓
       [Dropout + Add & Norm]     ← Residual Connection + LayerNorm
        ↓
       Output
    """

    def __init__(
        self,
        params: TransformerParams,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.mha = MultiHeadAttention(params, name="attention")

        assert params.mha_output_dim is not None
        # Feed Forward Network (optional)
        self.ffn = None
        if params.internal_dim > 0:
            self.ffn = PWFeedForward(
                output_dim=params.mha_output_dim,
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
        self.layer_norm_1 = nn.LayerNorm(params.mha_output_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(params.mha_output_dim, eps=1e-6)

        self.dropout_1 = nn.Dropout(params.dropout_rate)
        self.dropout_2 = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [B, S, D], D_model => Transformer hidden size（embedding dim）
        mask: Broadcastable to [B, H, S, S] (For MHA)
        """
        # [B, S, D] -> [B, S_q, D_mha_out], [B, H, S_q, S_k]
        attn_output, attention_weights = self.mha(x, x, x, mask)

        self.attention_weights = attention_weights
        attn_output = self.dropout_1(attn_output)
        # D_model is the transformer hidden size（embedding dim）
        # D_model = D_mha_out
        #  attn_output ∈ ℝ[B, S, D_model]
        #  x           ∈ ℝ[B, S, D_model]
        out_1 = self.layer_norm_1(x + attn_output)

        if self.ffn is None:
            return out_1
        # [B, S_q, D_mha_out] -> [B, S_q, D_mha_out]
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        return self.layer_norm_2(out_1 + ffn_output)


class DecoderLayer(nn.Module):
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

    def __init__(
        self,
        params: TransformerParams,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.mha_1 = MultiHeadAttention(params, name="attention_1")
        self.mha_2 = MultiHeadAttention(params, name="attention_2")

        assert params.mha_output_dim is not None
        # Feed Forward Network (optional)
        self.ffn = None
        if params.internal_dim > 0:
            self.ffn = PWFeedForward(
                output_dim=params.mha_output_dim,
                internal_dim=params.internal_dim,
                name="fc",
            )

        self.layer_norm_1 = nn.LayerNorm(params.mha_output_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(params.mha_output_dim, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(params.mha_output_dim, eps=1e-6)

        self.dropout_1 = nn.Dropout(params.dropout_rate)
        self.dropout_2 = nn.Dropout(params.dropout_rate)
        self.dropout_3 = nn.Dropout(params.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> ThreeTensors:
        """
        x: [B, S, D]
        enc_output: [B, S_q, D_mha_out]
        *_mask: Broadcastable to [B, H, S, S] (For MHA)
        """
        # Masked Self-Attention: `look_ahead_mask` -> Mask future tokens
        # [B, S, D] -> [B, S_q, D_mha_out]
        attn_1, attn_weights_block_1 = self.mha_1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout_1(attn_1)
        # Residual Connection + LayerNorm
        # D_model is the transformer hidden size（embedding dim）
        # D_model = D_mha_out
        #  attn_1 ∈ ℝ[B, S, D_model]
        #  x      ∈ ℝ[B, S, D_model]
        out_1 = self.layer_norm_1(attn_1 + x)

        # Encoder–Decoder Attention (Cross Attention)
        # The Decoder "queries" the output of the Encoder based on the current generation status
        attn_2, attn_weights_block_2 = self.mha_2(
            enc_output,   # v
            enc_output,   # k
            out_1,        # q
            padding_mask, # padding_mask -> Prevent focusing on padding in the Encoder `enc_output`
        )
        attn_2 = self.dropout_2(attn_2)
        out_2 = self.layer_norm_2(attn_2 + out_1)

        if self.ffn is not None:
            # [B, S_q, D_mha_out] -> [B, S_q, D_mha_out]
            ffn_output = self.ffn(out_2)
            ffn_output = self.dropout_3(ffn_output)
            out_3 = self.layer_norm_3(ffn_output + out_2)
        else:
            out_3 = out_2

        return out_3, attn_weights_block_1, attn_weights_block_2


class Encoder(nn.Module):
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
        layer_dropout_prob: float = 0.,
        skip_last_nonlinearity: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.num_layers = params.num_layers
        self.layer_dropout_prob = layer_dropout_prob

        layers = [
            EncoderLayer(params, name=f"layer_{i+1}")
            for i in range(params.num_layers - 1)
        ]
        if skip_last_nonlinearity:
            params = dataclasses.replace(
                params, activation_fn=None
            )
        layers.append(EncoderLayer(params, name=f"layer_{params.num_layers}"))
        self.enc_layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, S, D]
        mask: Broadcastable to [B, H, S, S] (For MHA)
        """
        # Layer dropout only takes effect during the training stage
        for layer in self.enc_layers:
            if self.training and torch.rand(()) < self.layer_dropout_prob:
                # Skip entire layer
                continue
            x = layer(x, mask)

        # [B, S_q, D_mha_out]
        return x


class Decoder(nn.Module):
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
        super().__init__()

        self.name = name
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

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        look_ahead_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, S, D]
        enc_output: [B, S_q, D_mha_out]
        *_mask: Broadcastable to [B, H, S, S] (For MHA)
        """
        for i in range(self.num_layers):
            x, _, _ = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )
        return x


class EncoderDecoderModel(nn.Module):
    """Transformer model."""

    def __init__(
        self,
        params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None
    ):
        super().__init__()

        self.name = name
        self.encoder = Encoder(params, name="encoder")
        self.decoder = Decoder(
            params,
            name="decoder",
            skip_last_nonlinearity=skip_last_nonlinearity,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [batch_size, seq_len, hidden_dim] ← Normal conditions
        # [seq_len, hidden_dim]             ← NO batch dimension
        single_sequence = len(x.shape) == 2
        if single_sequence:
            # [S, D] → [1, S, D]
            x = x.unsqueeze(dim=0)

        encoding = self.encoder(x, mask=mask)
        output: torch.Tensor = self.decoder(
            x,
            encoding,
            look_ahead_mask=mask,
            padding_mask=mask,
        )
        if single_sequence:
            # [1, S, D] → [S, D]
            output = output.squeeze(dim=0)

        return output


class EncoderModel(nn.Module):
    """Transformer model."""

    def __init__(
        self,
        params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.encoder = Encoder(
            params,
            name="encoder",
            skip_last_nonlinearity=skip_last_nonlinearity,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [batch_size, seq_len, hidden_dim]    ← Normal conditions
        # [seq_len, hidden_dim]                ← NO batch dimension
        single_sequence = len(x.shape) == 2
        if single_sequence:
            # [S, D] → [1, S, D]
            x = x.unsqueeze(dim=0)

        output: torch.Tensor = self.encoder(x, mask=mask)
        if single_sequence:
            # [1, S, D] → [S, D]
            output = output.squeeze(dim=0)
        return output


class SeparateEncoderDecoderModel(nn.Module):
    """Model using encoder for samples and decoder for weights."""

    def __init__(
        self,
        encoder_params: TransformerParams,
        decoder_params: TransformerParams,
        skip_last_nonlinearity=False,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.encoder = Encoder(encoder_params, name="encoder")
        self.decoder = Decoder(
            decoder_params,
            name="decoder",
            skip_last_nonlinearity=skip_last_nonlinearity,
        )

    def forward(
        self,
        sample_sequence: torch.Tensor,
        weight_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert mask is None

        # [batch_size, seq_len, hidden_dim]    ← Normal conditions
        # [seq_len, hidden_dim]                ← NO batch dimension
        single_sequence = len(sample_sequence.shape) == 2
        if single_sequence:
            # [S, D] → [1, S, D]
            sample_sequence = sample_sequence.unsqueeze(dim=0)
            weight_sequence = weight_sequence.unsqueeze(dim=0)

        encoding: torch.Tensor = self.encoder(sample_sequence)
        output: torch.Tensor = self.decoder(weight_sequence, encoding)
        if single_sequence:
            output = output.squeeze(dim=0)
        return output
