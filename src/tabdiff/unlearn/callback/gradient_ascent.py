import omegaconf
import torch

from typing import Callable, Optional, Union
from .template import UnlearnCallback

NUMERICAL_REPR = "numerical"


class GradientAscent(UnlearnCallback):
    """
    Class implementation of Gradient-Ascent as a loss-function. The callback should be provided to a training, so to..


    Notes:
        Currently, this implementation only supports 'vertical' unlearning, i.e., feature-wise unlearning.
    """

    def __init__(self, column: Optional[int], *, numerical: bool = True, **_):
        super().__init__(numerical=numerical)
        self.column: Optional[int] = column
        self.feature_index = None

    def init_mask_(self, D_yn: torch.FloatTensor):
        # feature_dim = D_yn.shape[-1]
        self.feature_index = torch.tensor(
            [self.column], dtype=torch.long, device=D_yn.device
        )

    def forward(
        self,
        D_yn: torch.FloatTensor,
        x0: Union[torch.FloatTensor, torch.LongTensor],
        xt: torch.FloatTensor,
        weight: torch.FloatTensor,
        *args,
        loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        ordinary_objective = self.compute_base_loss(D_yn, x0, loss_fn=loss_fn)
        if self.column is None:
            # In case the column is a None type, we return ordinar objective...
            return torch.zeros_like(ordinary_objective)

        # In case we are unlearning, first make sure to init
        if self.feature_index is None:
            self.init_mask_(D_yn)

        if self.feature_index is None:
            # If no feature is 'ascended', we simply return 0.
            # However, let it require gradients, to prevent maybe edge cases.
            no_loss: torch.FloatTensor = torch.zeros(
                (D_yn.shape[0], 1),
                device=D_yn.device,
                dtype=torch.FloatTensor,
                requires_grad=True,
            )
            return no_loss
        ordinary_objective = self.compute_base_loss(D_yn, x0, loss_fn=loss_fn)

        # TO
        # Flip the sign of the objective to obtain neg-grad equivalent...
        ordinary_objective[
            torch.arange(ordinary_objective.size(0)), self.feature_index
        ] = (
            -1.0
            * ordinary_objective[
                torch.arange(ordinary_objective.size(0)), self.feature_index
            ]
        )
        all_feature = torch.arange(
            ordinary_objective.size(1), device=self.feature_index.device
        )
        zero_mask = ~torch.isin(all_feature, self.feature_index)
        zero_out_indices = all_feature[zero_mask]
        ordinary_objective[
            torch.arange(ordinary_objective.size(0)),
            unsqueeze_like(zero_out_indices, ordinary_objective),
        ] = 0.0

        return ordinary_objective

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype: str, *args, **kwargs):
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        # Get columns
        dtype_columns = list(
            filter(lambda col: col.dtype == dtype, dataset_config.columns)
        )
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}

        # Note, column may not exist in own dtype
        column_index = lut.get(int(unlearn_config.column), None)

        is_numerical = dtype == "numerical"
        return GradientAscent(
            column_index,
            *args,
            numerical=is_numerical,
            **kwargs,
        )


def unsqueeze_like(
    in_tensor: torch.Tensor,
    like_tensor: torch.Tensor,
) -> torch.Tensor:
    dim_diff = like_tensor.ndim - in_tensor.ndim
    if dim_diff < 0:
        raise ValueError(f"Cannot squeeze from {in_tensor.ndim} to {like_tensor.ndim}.")
    if dim_diff > 0:
        # Unpack with the number of differences, adapted from Pytorch Forecasting's unsqueeze_like.
        in_tensor = in_tensor[(...,) + (None,) * dim_diff]
    return in_tensor
