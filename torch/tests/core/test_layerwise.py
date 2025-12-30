"""Tests for `layerwise.py`."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypertransformer.core import common_ht
from hypertransformer.core import layerwise
from hypertransformer.core import layerwise_defs # pylint:disable=unused-import


def make_layerwise_model_config():
    """Makes 'layerwise' model config."""
    return common_ht.LayerwiseModelConfig()
