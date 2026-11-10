from typing import Any, Dict

import numpy as np
import torch
from collections.abc import Mapping


def align_collator(
    features: list[Dict[str, np.ndarray | torch.Tensor]],
) -> dict[str, torch.FloatTensor]:
    """
    Adapted from huggingface transformers, adapted to TabDiff diffusion like models. Performs aligned mappi n
    https://github.com/huggingface/transformers/blob/e7d351cebad5f6dcdd169b0c034fdee0a000e6a9/src/transformers/data/data_collator.py#L75-L97
    Args:
        features (List[Dict[str, np.ndarray | torch.Tensor | torch.Tensor])): Batch of individual samples to be collated
            to a Dict of Tensor.

    Returns:

    """

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]

    batch = {}

    label = "sample"
    alt_label = "perturbed"

    expected_keys = [label, alt_label]
    dtype = torch.float
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    for key in expected_keys:
        batch[key] = torch.stack([f[key] for f in features]).to(dtype)
    return batch
