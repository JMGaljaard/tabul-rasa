import logging

import numpy as np
import omegaconf
import torch

from .feature import BaseFeatureAugment
from typing import Optional, Tuple, Union, List, Any, Dict, Callable
from tabdiff.utils import load_dataset_info


class RemapTransform:
    """
    Transformation object, to map feature-based tensors to a Shuffled distribution, while retaining the marginal
    distribution fo the original,
    """

    def __init__(
        self,
        encoded_data_indices: torch.LongTensor,
        stacked_resampled_values: Optional[torch.Tensor] = None,
    ):
        self.encoded_data_indices = encoded_data_indices
        self.resampled_values = stacked_resampled_values

    def __call__(self, sample, sample_index=-1):

        original_shape = sample.shape
        if len(sample.shape) == 1:
            # Make sure everything can be done as a batched operator.
            sample = sample.reshape((1, -1))
        # Ensure the permutations do not influence underlying data tensor.
        resampled = sample.clone().detach()
        if self.resampled_values is not None:
            resampled[:, self.encoded_data_indices] = self.resampled_values[
                :, sample_index
            ]
        else:
            # Otherwise, we use the bath's distribtuion, to encode, following the supplementary materials of
            # the FeatureShuffling paper.
            original_distribution = resampled[:, self.encoded_data_indices]
            resampled_indices = torch.randperm(original_distribution.numel())
            resampled[:, self.encoded_data_indices] = original_distribution[
                resampled_indices
            ]

        return resampled.reshape(original_shape), sample_index


def static_mapping_transform(
    resampled_values: torch.Tensor,
    indices: np.ndarray[int],
) -> Callable[
    [torch.FloatTensor, torch.LongTensor],
    Tuple[torch.FloatTensor, torch.LongTensor],
]:
    """
    Targeted Model input domain transformation for features that lie in the numerical domain.
    Args:
        margin (float): Margin that should be achieved by re-sampling
        indices (torch.LongTensor): Encoded data's indices to transform (noise)

    Returns:
        callable: Function implementing the (random) augmentation strategy.
    """
    logging.warning("Numerical transform assumes Gaussian prior...")
    # Define indices that require encodings.
    encoded_data_indices = torch.from_numpy(indices).to(torch.long)
    stacked_resampled_values = torch.stack(resampled_values)

    return RemapTransform(encoded_data_indices, stacked_resampled_values)


def construct_static_transforms(
    data_to_representation_lut: dict[str | int, str | int],
    encoded_data: torch.FloatTensor,
    numerical_offset: int,
    output_dist_offset: bool | Any,
    target_columns: List[Any] | Any,
    target_dtypes: List[Any] | Any,
    unlearn_seed,
) -> List[
    Callable[
        [torch.FloatTensor, torch.LongTensor],
        Tuple[torch.FloatTensor, torch.LongTensor],
    ]
]:
    """

    Args:
        data_to_representation_lut ():
        encoded_data ():
        numerical_offset ():
        output_dist_offset (bool): Whether unlearning is concerned with regression (True) or categorical bin/mult (False
            ).
        target_columns ():
        target_dtypes ():
        unlearn_seed ():

    Returns:

    """

    generator = torch.Generator()
    generator.manual_seed(
        unlearn_seed,
    )
    concat_index, split_index = construct_data_indices(
        data_to_representation_lut,
        numerical_offset,
        output_dist_offset,
        target_columns,
        target_dtypes,
    )
    resampled_distributions, targeted_columns = [], []
    for numpy_index, torch_index in zip(concat_index, split_index):
        # Note: this is bad practise.
        original_distribution = encoded_data[:, numpy_index].detach().clone()

        # Nice part is that every subset of columns (minus the
        permuted_distribution_index = torch.randperm(
            original_distribution.numel(),
            generator=generator,
        )
        permuted_distribution = original_distribution[permuted_distribution_index]

        resampled_distributions.append(permuted_distribution)
        # The *encoded* index should be used to permute the distribution
        targeted_columns.append(torch_index)

    # Define transformations to perform on the encoded data-domain (diffusion model input)
    targeted_columns = np.array(targeted_columns).astype(int)
    transforms = [
        static_mapping_transform(resampled_distributions, targeted_columns),
    ]
    return transforms


def construct_data_indices(
    data_to_representation_lut: dict[str | int, str | int],
    numerical_offset: int,
    output_dist_offset: bool | Any,
    target_columns: list[Any] | Any,
    target_dtypes: list[Any] | Any,
) -> tuple[list[Any], list[Any]]:
    """
    Funciton to map 'disk' ordering of column to their 'encoded' counter-parts.
    Args:
        data_to_representation_lut ():
        numerical_offset (): Number of numerical columns (regresssion + num + int)
        output_dist_offset (): Idk
        target_columns (): Good question.
        target_dtypes (): Real ship.

    Returns:
        List[int]: Encoding of indices in 'flat' represtnation, where [int, cat] attributes are concatenated.
        List[int]: Encodings of indices in 'split' represetnation, where either 'int' or 'cat' is used.
    """
    numpy_indices, torch_indices = [], []

    for data_index, dtype in zip(target_columns, target_dtypes):
        numpy_index = -1
        torch_index = -1
        match dtype:
            case _ if dtype in {"categorical", "numerical"}:
                # 0. We assume that we won't have a target being unlearned

                # if is regression 'dataset', the target is prepended. Either in the numerical case, or in the dis-
                # crete case.
                # if regression, IDK, this preprocessing is x
                numpy_index = data_to_representation_lut[data_index] + (
                    not output_dist_offset
                )
                if numpy_index > max(data_to_representation_lut.values()):
                    # In case we are targetting the target, we must wrap around to 0.
                    if output_dist_offset:
                        numpy_index = 0
                    else:
                        # In case of categorical dataset, target prepended here.
                        numpy_index = numerical_offset

                # Now, we have the problem that we don't know the encoded index properly
                # The *encoded* index should be used to permute the distribution
                if dtype == "numerical":
                    torch_index = numpy_index
                else:
                    # If classification, then we need to 'shift', because the input is split in
                    #   two tensors.
                    torch_index = numpy_index - numerical_offset

            case _:
                raise NotImplementedError(
                    f"Data of type: {dtype} is not supported, at index {data_index}."
                )
        numpy_indices.append(numpy_index)
        torch_indices.append(torch_index)

    return numpy_indices, torch_indices


class PerturbAugment(BaseFeatureAugment):
    """
    Implements 'mapped' (deterministic) transforms on pre-encoded data (diffusion model domain).



    Notes:
        This only exists, because the data-handling by tabdiff is rather inane.
        Assumes that the data index (requested atleast) is available.
    """

    @classmethod
    def _from_config(
        cls,
        config: omegaconf.OmegaConf | omegaconf.DictConfig,
        encoded_data: torch.FloatTensor,
        *args,
        info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """

        Args:
            config (): Experiment configuration object.
            *args ():
            info (): Dataset meta-data.
            indx_to_perturb (): Per-column index alternative per
            **kwargs ():

        Returns:
            RandomPerturbAugment instance configured with requested augmentation strategy.
        """
        # 1. Get dataset info to find mapping after pre-processing
        #   (Note, this is regarded)
        info = info or load_dataset_info(config.dataset.name)
        unlearn_seed = config.unlearn.seed
        # Get lut offset in case there is a numerical column/s
        data_to_representation_lut: Optional[Dict[Union[str, int], Union[str, int]]] = (
            info.get("idx_mapping", None)
        )
        assert (
            data_to_representation_lut is not None
        ), "Are you refactoring the processing pipeline? Metadata object is broken"
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

        # Create static transforms used in
        output_dist_offset = info.get("task_type") == "regression"
        transforms = construct_static_transforms(
            data_to_representation_lut,
            encoded_data,
            numerical_offset,
            output_dist_offset,
            target_columns,
            target_dtypes,
            unlearn_seed,
        )

        return PerturbAugment(
            transforms,
        )
