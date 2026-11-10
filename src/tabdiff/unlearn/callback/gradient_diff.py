import omegaconf
import torch

from typing import Callable, Optional, Union


from .template import UnlearnCallback

NUMERICAL_REPR = "numerical"


class GradientDiff(UnlearnCallback):
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
            return ordinary_objective

        if self.feature_index is None:
            self.init_mask_(D_yn)

        # Flip the sign of the objective to obtain neg-grad equivalent...
        ordinary_objective[
            torch.arange(ordinary_objective.size(0)), self.feature_index
        ] = (
            -1.0
            * ordinary_objective[
                torch.arange(ordinary_objective.size(0)), self.feature_index
            ]
        )

        return ordinary_objective

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype: str, *args, **kwargs):
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        # Get columns with same dtype
        dtype_columns = list(
            filter(lambda col: col.dtype == dtype, dataset_config.columns)
        )
        # Build lut
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}

        # TODO: Make support for *indices*
        # Retrieve dataset index to 'encoded' index
        maybe_column_index = lut.get(int(unlearn_config.column), None)

        is_numeric = dtype == "numerical"
        return GradientDiff(
            maybe_column_index,
            *args,
            numerical=is_numeric,
            **kwargs,
        )
