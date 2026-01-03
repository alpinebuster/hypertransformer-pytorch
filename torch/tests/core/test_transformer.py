"""Tests for `transformer.py`."""

import pytest
import torch

from hypertransformer.core.transformer import attention, PWFeedForward, \
    MultiHeadAttention, TransformerParams, \
    EncoderDecoderModel, EncoderModel, SeparateEncoderDecoderModel, \
    EncoderLayer, DecoderLayer


# ------------------------------------------------------------
#   Function `attention` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S_q,S_k,d_qk,d_v",
    [
        (1, 1, 1, 1, 4, 4),
        (2, 4, 3, 5, 8, 6),
        (4, 8, 7, 7, 16, 8),
    ],
)
def test_attention_func(B, H, S_q, S_k, d_qk, d_v):
    q = torch.randn(B, H, S_q, d_qk)
    k = torch.randn(B, H, S_k, d_qk)
    v = torch.randn(B, H, S_k, d_v)

    context, attn = attention(q, k, v)

    assert context.shape == (B, H, S_q, d_v)
    assert attn.shape == (B, H, S_q, S_k)


# ------------------------------------------------------------
#   Class `PWFeedForward` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,internal_dim,output_dim",
    [
        (2, 4, 16, 32, 16),
        (1, 8, 32, 64, 32),
        (4, 1, 64, 128, 64),
        (2, 10, 128, 256, 128),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [torch.relu, None],
)
def test_pwfeedforward(
    batch_size,
    seq_len,
    input_dim,
    internal_dim,
    output_dim,
    activation,
):
    model = PWFeedForward(
        output_dim=output_dim,
        internal_dim=internal_dim,
        activation=activation,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    y: torch.Tensor = model(x)

    assert y.shape == (batch_size, seq_len, output_dim)


# ------------------------------------------------------------
#   Class `MultiHeadAttention` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,s_q,s_k,d_qk,d_v,heads",
    [
        (2, 4, 4, 32, 32, 1),    # Sinle head self-attention
        (2, 8, 8, 64, 64, 4),    # Multi heads self-attention
        (1, 5, 7, 128, 128, 8),  # Multi heads cross-attention
    ],
)
@pytest.mark.parametrize(
    "use_mask",
    [True, False],
)
@pytest.mark.parametrize(
    "d_model,mha_output_dim",
    [
        (128, 128),
        (256, 256),
        (512, 512),
    ],
)
def test_multi_head_attention_shapes(
    batch_size,
    s_q,
    s_k,
    d_qk,
    d_v,
    heads,
    use_mask,
    d_model,
    mha_output_dim,
):
    x_q = torch.randn(batch_size, s_q, d_model)
    x_k = torch.randn(batch_size, s_k, d_model)
    x_v = torch.randn(batch_size, s_k, d_model)

    if use_mask:
        # shape: [B, 1, 1, S_k] -> [B, H, S_q, S_k]
        # The two 1s in the middle are for broadcasting
        mask = torch.ones(batch_size, 1, 1, s_k)
        mask[:, :, :, -1] = 0  # Mask out the last token
    else:
        mask = None

    params = TransformerParams(
        query_key_dim=d_qk,
        value_dim=d_v,
        internal_dim=128,
        num_layers=1,
        mha_output_dim=mha_output_dim,
        heads=heads,
    )

    mha = MultiHeadAttention(params=params)

    output: torch.Tensor
    attn_weights: torch.Tensor
    output, attn_weights = mha(x_v, x_k, x_q, mask)

    assert output.shape == (batch_size, s_q, params.mha_output_dim)
    assert attn_weights.shape == (batch_size, heads, s_q, s_k)


# ------------------------------------------------------------
#   Class `EncoderLayer` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "query_key_dim,value_dim,num_layers,heads,internal_dim",
    [
        (128,  64, 1, 4, 1024), # params.internal_dim > 0  -> Has FFN
        (256, 128, 2, 8, 0),    # params.internal_dim <= 0 -> NO FFN
    ],
)
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (2, 43),
        (1, 82),
    ],
)
@pytest.mark.parametrize(
    "d_model,mha_output_dim",
    [
        (128, 128),
        (256, 256),
        (512, 512),
    ],
)
def test_encoder_layer(
    query_key_dim: int,
    value_dim: int,
    num_layers: int,
    heads: int,
    internal_dim: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    mha_output_dim: int,
):
    torch.manual_seed(0)

    params = TransformerParams(
        query_key_dim=query_key_dim,
        value_dim=value_dim,
        mha_output_dim=mha_output_dim,
        internal_dim=internal_dim,
        num_layers=num_layers,
        heads=heads,
        dropout_rate=0.1,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    # mask: [B, 1, 1, S]（broadcastable）
    mask = torch.ones(batch_size, 1, 1, seq_len)

    assert params.mha_output_dim is not None
    layer = EncoderLayer(params=params)
    output = layer(x, mask=mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, seq_len, params.mha_output_dim)

    # Attention weights should exist
    assert hasattr(layer, "attention_weights")
    attn: torch.Tensor = layer.attention_weights
    assert attn.shape == (batch_size, heads, seq_len, seq_len)


# ------------------------------------------------------------
#   Class `DecoderLayer` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "query_key_dim,value_dim,num_layers,heads,internal_dim",
    [
        (128,  64, 1, 4, 1024), # params.internal_dim > 0  -> Has FFN
        (256, 128, 2, 8, 0),    # params.internal_dim <= 0 -> NO FFN
    ],
)
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (2, 43),
        (1, 82),
    ],
)
@pytest.mark.parametrize(
    "d_model, mha_output_dim",
    [
        (128, 128),
        (256, 256),
        (512, 512),
    ],
)
def test_decoder_layer(
    query_key_dim: int,
    value_dim: int,
    num_layers: int,
    heads: int,
    internal_dim: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    mha_output_dim: int,
):
    torch.manual_seed(0)

    params = TransformerParams(
        query_key_dim=query_key_dim,
        value_dim=value_dim,
        mha_output_dim=mha_output_dim,
        internal_dim=internal_dim,
        num_layers=num_layers,
        heads=heads,
        dropout_rate=0.1,
    )
    # target sequence (decoder input)
    # D_mha_out = d_model
    x = torch.randn(batch_size, seq_len, d_model)
    # encoder output -> [B, S_q, D_mha_out]
    enc_output = torch.randn(batch_size, seq_len, d_model)

    # look-ahead mask: [B, 1, S, S]
    look_ahead_mask = torch.tril(
        torch.ones(seq_len, seq_len)
    ).unsqueeze(dim=0).unsqueeze(dim=1)
    look_ahead_mask = look_ahead_mask.expand(batch_size, 1, seq_len, seq_len)
    # padding mask: [B, 1, 1, S]
    padding_mask = torch.ones(batch_size, 1, 1, seq_len)

    layer = DecoderLayer(params)
    out: torch.Tensor
    attn_w1: torch.Tensor
    attn_w2: torch.Tensor
    out, attn_w1, attn_w2 = layer(
        x,
        enc_output,
        look_ahead_mask=look_ahead_mask,
        padding_mask=padding_mask,
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, seq_len, d_model)

    # self-attention weights
    assert attn_w1.shape == (batch_size, heads, seq_len, seq_len)

    # cross-attention weights
    assert attn_w2.shape == (batch_size, heads, seq_len, seq_len)


# ------------------------------------------------------------
#   Class `EncoderDecoderModel` Tests
# ------------------------------------------------------------


@pytest.fixture(params=[
    {
        "mha_output_dim": 512,
        "query_key_dim": 32,
        "value_dim": 32,
        "internal_dim": 1024,
        "num_layers": 2,
        "heads": 8,
    },
    {
        "mha_output_dim": 256,
        "query_key_dim": 64,
        "value_dim": 64,
        "internal_dim": 512, # params.internal_dim > 0  -> Has FFN
        "num_layers": 4,
        "heads": 4,
    },
    {
        "mha_output_dim": 256,
        "query_key_dim": 64,
        "value_dim": 64,
        "internal_dim": 0, # params.internal_dim <= 0 -> NO FFN
        "num_layers": 4,
        "heads": 4,
    },
])
def transformer_params(request):
    return TransformerParams(**request.param)

@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        # hidden_dim == mha_output_dim
        (2, 8), # [batch_size, seq_len, hidden_dim] ← Normal conditions
        (None, 8), # [seq_len, hidden_dim] ← NO batch dimension
    ],
)
def test_encoderdecodermodel(
    transformer_params: TransformerParams,
    batch_size: int,
    seq_len: int,
):
    torch.manual_seed(0)

    assert transformer_params.mha_output_dim is not None
    if batch_size is None:
        x = torch.randn(seq_len, transformer_params.mha_output_dim)
        expected_shape = (seq_len, transformer_params.mha_output_dim)
    else:
        x = torch.randn(batch_size, seq_len, transformer_params.mha_output_dim)
        expected_shape = (batch_size, seq_len, transformer_params.mha_output_dim)

    model = EncoderDecoderModel(transformer_params)
    model.eval()
    out: torch.Tensor = model(x)

    assert out.shape == expected_shape
    assert not torch.isnan(out).any()


# ------------------------------------------------------------
#   Class `EncoderModel` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (2, 8), # [batch_size, seq_len, hidden_dim] ← Normal conditions
        (None, 8), # [seq_len, hidden_dim] ← NO batch dimension
    ],
)
def test_encodermodel(
    transformer_params: TransformerParams,
    batch_size: int,
    seq_len: int,
):
    torch.manual_seed(0)

    assert transformer_params.mha_output_dim is not None
    if batch_size is None:
        x = torch.randn(seq_len, transformer_params.mha_output_dim)
        expected_shape = (seq_len, transformer_params.mha_output_dim)
    else:
        x = torch.randn(batch_size, seq_len, transformer_params.mha_output_dim)
        expected_shape = (batch_size, seq_len, transformer_params.mha_output_dim)

    model = EncoderModel(transformer_params)
    model.eval()
    out: torch.Tensor = model(x)

    assert out.shape == expected_shape
    assert not torch.isnan(out).any()


# ------------------------------------------------------------
#   Class `SeparateEncoderDecoderModel` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (2, 8), # [batch_size, seq_len, hidden_dim] ← Normal conditions
        (None, 8), # [seq_len, hidden_dim] ← NO batch dimension
    ],
)
def test_separateencoderdecoder_model(
    transformer_params: TransformerParams,
    batch_size: int,
    seq_len: int,
):
    torch.manual_seed(0)

    encoder_params = transformer_params
    decoder_params = transformer_params

    assert transformer_params.mha_output_dim is not None
    if batch_size is None:
        x = torch.randn(seq_len, transformer_params.mha_output_dim)
        weight_sequence = torch.randn(seq_len*2, transformer_params.mha_output_dim)
        expected_shape = (seq_len*2, transformer_params.mha_output_dim)
    else:
        x = torch.randn(batch_size, seq_len, transformer_params.mha_output_dim)
        weight_sequence = torch.randn(batch_size, seq_len*2, transformer_params.mha_output_dim)
        expected_shape = (batch_size, seq_len*2, transformer_params.mha_output_dim)

    model = SeparateEncoderDecoderModel(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
    )
    output = model(x, weight_sequence)

    assert isinstance(output, torch.Tensor)
    assert output.shape == expected_shape
