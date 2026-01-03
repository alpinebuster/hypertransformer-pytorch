"""Common basic utilities and Transformer adapters."""

import functools
import glob
import math
import os
from typing import Callable, Optional, Iterable, Union, Any

from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hypertransformer.core import transformer

TransformerParamsFn = Callable[[int], transformer.TransformerParams]


# ------------------------------------------------------------
#   TransformerIO
# ------------------------------------------------------------


class _TransformerIO(nn.Module):
    """Encoder and decoder interfacing with the Transformer.

    1. SimpleTransformerIO
       W is query, which is read outside the Transformer

        [ L   | I... ]  --> Transformer               --> [K | V] (memory)
        [ W_i | 0..0 ]  --> Attention, dot(W,K^T) × V --> decoded_weight

    2. SeparateTransformerIO
       W is a token, but it is not in the same sequence as L/I

        [ W_i | 0..0 ]  --> Transformer --> [ W_i | decoded_weight_i ]
        [ L   | I... ]  --> Transformer --> [ sample_output ]

    3. JointTransformerIO
       Both W and L/I are tokens and can pay attention to each other

        [[ W_i | 0..0 ]     [ W_i | decoded_weight_i ]
             ...        -->            ...
         [ L   | I... ]]    [ sample_output          ]

    Notation (vector-level representation):

       weight token = [ weight_emb |   zeros    ]
                       ←-- D_w --→  ←-- I_w --→

       label token  = [ label_emb  | images_emb ]
                       ←-- D_l --→  ←-- I_l --→

       a) W_i = [ w1 , w2 , ... ]
           Weight embedding for the i-th weight block.
           This vector represents the identity or query of a learnable weight.

       b) L   = [ l1 , l2 , ... ]
           Label embedding. A continuous vector representation of a discrete class label.
    """

    def __init__(
        self,
        num_labels: int,
        num_weights: int,
        weight_block_size: int, # I_w
        embedding_dim: int = 8, # D_w
        weight_embedding_dim: Optional[int] = None, # D_l
    ):
        super().__init__()

        self.num_labels = num_labels
        self.num_weights = num_weights
        self.embedding_dim = embedding_dim

        if weight_embedding_dim is None:
            weight_embedding_dim = embedding_dim
        self.weight_embedding_dim = weight_embedding_dim
        self.weight_block_size = weight_block_size

        # One additional class is reserved for "no label" class
        # labels:         (num_samples,)
        #               ↓
        # encoded_labels: (num_samples, embedding_dim)
        self.label_embs: nn.Embedding = nn.Embedding(
            num_embeddings=num_labels + 1,
            embedding_dim=embedding_dim,
        )
        # TensorFlow defaults to a normal distribution with stddev=1.0, 
        # while PyTorch defaults to a uniform distribution for `nn.Embedding` initialization
        nn.init.normal_(self.label_embs.weight, mean=0.0, std=1.0)

        # Weight embeddings are independent parameters (not an embedding table)
        self.weight_embs: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(torch.randn(weight_embedding_dim))
                for _ in range(num_weights)
            ]
        )
        for p in self.weight_embs:
            nn.init.normal_(p, mean=0.0, std=1.0)

    def _encode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Generates label encodings.
        labels: LongTensor [batch]
        returns: FloatTensor [batch, embedding_dim]

        e.g. `tf.gather`
           label_embs = [
               [1.0, 0.0],   # 0
               [0.0, 1.0],   # 1
               [1.0, 1.0],   # 2
           ]
   
           labels = [2, 0]
   
           tf.gather(label_embs, labels, axis=0) == [
               [1.0, 1.0],   # index 2
               [1.0, 0.0],   # index 0
           ]
        """
        return self.label_embs(labels)

    def decode_weights(self, embeddings: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class SimpleTransformerIO(_TransformerIO):
    """Encoder and decoder interfacing with the Transformer."""

    def encode_samples(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generates Transformer inputs from input samples.

        Args:
           images: FloatTensor [batch, image_size, image_size, channels]
           labels: LongTensor  [batch]

        Returns: 
           FloatTensor [batch, label_dim + image_dim]
        """
        batch_size: int = images.shape[0]

        # Flatten image features
        #                                I_l
        images = images.view(batch_size, -1)
        encoded_labels: torch.Tensor = self._encode_labels(labels)
        """
        encoded_labels: [batch_size, label_dim]
        images:         [batch_size, image_dim]
                            ↓
        [batch_size, label_dim + image_dim]
        """
        return torch.cat([encoded_labels, images], dim=-1)

    def extend_label_mask(self, label_mask: torch.Tensor) -> torch.Tensor:
        return label_mask

    def decode_weights(
        self,
        embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Generates weight patches from Transformer outputs.

        Args:
           embeddings: FloatTensor [..., D_w + I_w]

        Returns:
           List of decoded weight tensors, each of shape [I_w]
        """
        weights: list[torch.Tensor] = []

        # Split transformer outputs into keys and values
        weight_keys: torch.Tensor = embeddings[..., :self.weight_embedding_dim]
        weight_values: torch.Tensor = embeddings[..., self.weight_embedding_dim:]
        for weight_emb in self.weight_embs:
            """
            mixture[i] = softmax( Σ_j weight_emb[j] * weight_keys[i, j] )
            decoded[j] = Σ_t mixture[t] * weight_values[t, j]
            """
            # Dot product over embedding dim
            # weight_emb:        [D_w]
            # weight_keys:       [..., D_w]
            # weight_values:     [..., I_w]
            # mixture logits:    [...]
            mixture = torch.einsum("j,...j->...", weight_emb, weight_keys)
            mixture = F.softmax(mixture, dim=-1)

            weights.append(torch.einsum("..., ...j->j", mixture, weight_values))

        return weights


class SeparateTransformerIO(SimpleTransformerIO):
    """IO for feeding samples into Encoder and getting weights from Decoder."""

    def encode_weights(self) -> torch.Tensor:
        """Generates weight patches from Transformer outputs.

        Returns:
           tokens: FloatTensor [num_weights, D_w + I_w]
        """
        tokens = []
        for i in range(self.num_weights):
            weight_emb: torch.Tensor = self.weight_embs[i]  # [D_w]
            weight_zeros: torch.Tensor = torch.zeros(
                self.weight_block_size,
                device=weight_emb.device,
                dtype=weight_emb.dtype,
            )  # [I_w]

            token: torch.Tensor = torch.cat([weight_emb, weight_zeros], dim=-1)
            tokens.append(token)
        return torch.stack(tokens, dim=0)

    def decode_weights(
        self,
        embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Generates weight patches from Transformer outputs."""
        weights: list[torch.Tensor] = []

        for i in range(self.num_weights):
            weights.append(embeddings[i, self.weight_embedding_dim:])
        return weights


class JointTransformerIO(_TransformerIO):
    """Encoder and decoder interfacing with the Transformer."""

    def encode_samples(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generates Transformer inputs from input samples.

        Args:
           images: FloatTensor [B, ...]
           labels: LongTensor  [B]

        Returns:
           sequence: FloatTensor [num_weights + B, D_w + I_l]
        """        
        # Sample
        batch_size: int = images.shape[0]
        # Flatten image features
        #                                                   I_l
        images_flat: torch.Tensor = images.view(batch_size, -1)
        encoded_labels: torch.Tensor = self._encode_labels(labels)
        # [B, D_l + I_l]
        sequence: torch.Tensor = torch.concat([encoded_labels, images_flat], dim=-1)

        # Weight
        weight_sequence: Union[list[torch.Tensor], torch.Tensor] = []
        image_dim: int = images_flat.shape[1]
        for i in range(self.num_weights):
            weight_emb: torch.Tensor = self.weight_embs[i] # [D_w]
            zero_emb: torch.Tensor = torch.zeros(
                image_dim,
                dtype=sequence.dtype,
                device=sequence.device,
            ) # [I_l]

            weight_token: torch.Tensor = torch.cat([weight_emb, zero_emb], dim=-1)
            weight_sequence.append(weight_token)
        # [num_weights, D_w + I_l]
        weight_sequence = torch.stack(weight_sequence, dim=0)

        # Concatenate weight tokens before sample tokens
        # [num_weights + B, D_w + I_l]
        return torch.cat([weight_sequence, sequence], dim=0)

    def extend_label_mask(self, label_mask: torch.Tensor) -> torch.Tensor:
        """
        v---  num_weights  ---v\n
        [0, 0, 0, ..., 0, 0, 0]
                   ↓
        label_mask = [m0, m1, m2, ..., mk]
                   ↓
        [0, 0, 0, ..., 0, 0, 0, m0, m1, m2, ..., mk]\n
         ^--- num_weights ---^
        """
        weight_mask = torch.zeros(
            self.num_weights,
            dtype=label_mask.dtype,
            device=label_mask.device,
        )
        return torch.concat([weight_mask, label_mask], dim=-1)

    def decode_weights(
        self,
        embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Generates weight patches from Transformer outputs."""
        weights: list[torch.Tensor] = []
        for i in range(self.num_weights):
            weights.append(embeddings[i, self.weight_embedding_dim:])
        return weights


# ------------------------------------------------------------
#   Utils
# ------------------------------------------------------------


def _parse_label_spec(label_spec: str):
    """Parses label specification."""
    labels = []
    parts = label_spec.split(",")
    for part in parts:
        subset = part.split("-")
        if len(subset) == 1:
            labels.append(int(subset[0]))
        elif len(subset) == 2:
            labels.extend(range(int(subset[0]), int(subset[1]) + 1))
        else:
            raise ValueError("Wrong label specification format.")
    return labels

def parse_dataset_spec(dataset_spec: str):
    """Parses the dataset specification."""
    if ":" not in dataset_spec:
        return dataset_spec, None
    parts = dataset_spec.split(":")
    if len(parts) > 2:
        raise ValueError(
            "Wrong dataset specification format: should be a dataset name, or be "
            'of the format "dataset_name:1-10,20-30,40".'
        )
    dataset_spec, label_spec = parts
    return dataset_spec, _parse_label_spec(label_spec)


def nonlinearity(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "lrelu":
        return functools.partial(F.leaky_relu, negative_slope=0.1)
    else:
        raise ValueError(f"Unknown nonlinearity {activation}.")


def _extract_checkpoint_step(s: str) -> int:
    """
    Extract step from checkpoint filename with format `prefix-<step>.pth`.

    Example:
        model-1000.pth -> 1000
    """
    stem = s.rsplit(".", 1)[0]
    return int(stem.rsplit("-", 1)[1])

def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Find latest checkpoint in directory by step number."""
    all_checkpoints = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not all_checkpoints:
        return None

    latest = max(all_checkpoints, key=_extract_checkpoint_step)
    return latest

def latest_checkpoint(dir_or_checkpoint: str) -> Optional[str]:
    """
    Resolve checkpoint path.

    - If file exists → return it
    - If directory → find latest checkpoint
    """
    # That's actual checkpoint prefix.
    if os.path.isfile(dir_or_checkpoint):
        return dir_or_checkpoint

    if os.path.isdir(dir_or_checkpoint):
        return _find_latest_checkpoint(dir_or_checkpoint)

    return None


def load_variables(
    loc: str,
    var_list: Optional[Iterable[str]] = None,
    step: Optional[int] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Load variables from a PyTorch checkpoint.

    Args:
        loc: Checkpoint file or directory.
        var_list: Variable names to load.
        step: Optional step suffix.
        map_location: torch.load map_location.

    Returns:
        dict[name, Tensor]
    """
    # File or Directory → latest checkpoint
    path = latest_checkpoint(loc)
    if path is None:
        raise FileNotFoundError(f'No checkpoint available found at "{loc}"')

    loc = path
    if step is not None:
        base, ext = os.path.splitext(loc)
        loc = f"{base}-{step}{ext}"
    if not os.path.exists(loc):
        raise FileNotFoundError(f'Checkpoint not found: "{loc}"')

    logging.info(f"Loading from {loc}")
    ckpt: dict = torch.load(loc, map_location=map_location)
    state_dict: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)

    if var_list is None:
        return dict(state_dict)

    return {
        name: state_dict[name]
        for name in var_list
        if name in state_dict
    }


class MultiFileWriter:
    """Summary writer that supports writing to multiple files at once.

    Usage:
       ```py
       writer = MultiFileWriter("logs")

       writer.add_scalar("loss_1", 0.23, step, name="train")
       writer.add_scalars(
           "loss_2",
           {
               "train": 0.23,
               "val": 0.31,
           },
           global_step=10,
       )

       writer.flush()
       writer.close()
       ```
    """

    def __init__(self, logdir: str, **kwargs):
        self.summary_kwargs = dict(kwargs)
        self.logdir = logdir
        self.writers: dict[Optional[str], SummaryWriter] = {}

    def _get_writer(self, name: Optional[str] = None) -> SummaryWriter:
        if name not in self.writers:
            self.writers[name] = SummaryWriter(
                os.path.join(self.logdir, name or ""),
                **self.summary_kwargs,
            )
        return self.writers[name]

    def _normalize_tag(self, tag: str) -> str:
        """
        Converts xxx_1, xxx_2 -> xxx
        Keeps original tag if suffix is not numeric.
        """
        if "_" not in tag:
            return tag
        normalized_tag, value = tag.rsplit("_", 1)
        return normalized_tag if value.isdigit() else tag

    def add_scalar(
        self,
        tag: str,
        value: float,
        global_step: Optional[int] = None,
        name: Optional[str] = None
    ):
        tag = self._normalize_tag(tag)
        self._get_writer(name).add_scalar(tag, value, global_step)

    def add_scalars(
        self,
        scalars: dict[str, float],
        global_step: Optional[int] = None
    ):
        """
        Write multiple scalars to different sub-writers.
        """
        for name, value in scalars.items():
            self._get_writer(name).add_scalar(
                self._normalize_tag(name), value, global_step
            )

    def close(self):
        for each in self.writers.values():
            each.close()

    def flush(self):
        for each in self.writers.values():
            each.flush()


def same_pad_2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """
    TensorFlow-style SAME padding for 2D convolution.
    Supports stride > 1.

    Args:
       x -> [B, C, H, W]
    """

    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size

    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    if isinstance(dilation, int):
        dh, dw = dilation, dilation
    else:
        dh, dw = dilation

    _, _, h, w = x.shape

    # Effective kernel size after dilation
    # e.g. kernel_size = 3
    #  dilation=1 -> ● ● ●
    #  dilation=2 -> ● _ ● _ ●
    #       ↓
    #  Effective Size = 5
    kh_eff = (kh - 1) * dh + 1
    kw_eff = (kw - 1) * dw + 1

    # The goal of SAME padding
    out_h = math.ceil(h / sh)
    out_w = math.ceil(w / sw)

    # Target: out = ceil(in / stride)
    #             ↓
    # Actual: out = floor((in + pad_left + pad_right − effective_kernel) / stride) + 1
    #             ↓
    #     Why can "floor" be ignored? Because: 
    #       The design goal of SAME padding is "padding large enough to ensure that the output is at least `ceil(in/stride)`". 
    #       A little more padding will not affect the output size.
    #     out − 1 = (in + pad_total − effective_kernel) / stride, (pad_total = pad_left + pad_right)
    #             ↓
    # (in + pad_total − effective_kernel) = (out − 1) × stride
    #             ↓
    #         pad = (out - 1) * stride + effective_kernel - in
    pad_h = max((out_h - 1) * sh + kh_eff - h, 0)
    pad_w = max((out_w - 1) * sw + kw_eff - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

def print_gpu_detailed_info() -> None:
    logging.info("========== GPU / CUDA INFO ==========")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "<ALL>")}")
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"PyTorch CUDA Version: {torch.version.cuda}") # type: ignore[attr-defined]

    if not torch.cuda.is_available():
        logging.info("Running on CPU only")
        logging.info("====================================")
        return

    logging.info("Visible GPU Count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        total_mem = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3

        logging.info(f"\n--- GPU {i} ---")
        logging.info(f"Name: {props.name}")
        logging.info(f"Compute Capability: {props.major}.{props.minor}")
        logging.info(f"Total Memory: {total_mem:.2f} GB")
        logging.info(f"Memory Allocated: {allocated:.2f} GB")
        logging.info(f"Memory Reserved: {reserved:.2f} GB")
        logging.info(f"Multi-processor Count: {props.multi_processor_count}")
        logging.info(f"Max Threads per SM: {props.max_threads_per_multi_processor}")

    logging.info("====================================")
