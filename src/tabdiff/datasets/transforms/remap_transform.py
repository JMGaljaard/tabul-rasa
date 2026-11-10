from typing import Callable, Dict, Iterable, Optional, Union

import omegaconf
import torch
from torch.utils.data import default_collate

from .tensor_transform import BaseTensorTransform
from tabdiff.datasets.augment.feature_label import (
    RemapTransform,
    construct_data_indices,
)
from ...utils import load_dataset_info


class TensorRemapTransform(BaseTensorTransform):
    """
    Implements torch Transform function as defined in the supplementary materials of.
    """

    @classmethod
    def _from_config(
        cls,
        config: omegaconf.DictConfig,
        *args,
        collate_fn: Callable[[Iterable[torch.FloatTensor]], torch.FloatTensor],
        **kwargs,
    ):
        # Get dataset specific information / meta-data
        info = kwargs.get("info", None) or load_dataset_info(config.dataset.name)

        # Get lut offset in case there is a numerical column/s
        data_to_representation_lut: Optional[Dict[Union[str, int], Union[str, int]]] = (
            info.get("idx_mapping", None)
        )

        numerical_offset = len(info.get("num_col_idx", []))
        # Determine if margin is needed.

        # 2. Get columns to 'remove'
        unlearn_config = config.unlearn
        target_columns = (
            unlearn_config.columns
            if hasattr(unlearn_config, "columns")
            else [unlearn_config.column]
        )
        target_dtypes = (
            unlearn_config.dtypes
            if hasattr(unlearn_config, "dtypes")
            else [unlearn_config.dtype]
        )
        # Retrieve the encoded data-indices for the processin function.
        output_dist_offset = info.get("task_type") == "regression"
        encoded_data_indices, _ = construct_data_indices(
            data_to_representation_lut=data_to_representation_lut,
            numerical_offset=numerical_offset,
            output_dist_offset=output_dist_offset,
            target_columns=target_columns,
            target_dtypes=target_dtypes,
        )
        transform_fn = RemapTransform(
            encoded_data_indices=encoded_data_indices,
            stacked_resampled_values=None,
        )
        return TensorRemapTransform(
            encoded_data_indices,
            transform_fn,
            collate_fn=collate_fn,
        )

    def __init__(
        self,
        encoded_data_indices: torch.LongTensor,
        transform_fn: RemapTransform,
        *args,
        collate_fn: Callable[
            [Iterable[torch.FloatTensor]], torch.FloatTensor
        ] = default_collate,
        **kwargs,
    ):

        super().__init__(transform_fn, *args, collate_fn=collate_fn, **kwargs)
        self.encoded_data_indices = encoded_data_indices
