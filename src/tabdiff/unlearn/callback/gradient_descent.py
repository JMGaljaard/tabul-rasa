import omegaconf
import torch

from typing import Callable, Optional, Union
from .template import UnlearnCallback

NUMERICAL_REPR = "numerical"


class GradientDescent(UnlearnCallback):
    """
    Class implementation of Gradient-Descent as a loss-function.



    Notes:
        Currently, this only implements a default callback to make overall process easier. Mayb.e
    """

    def __init__(self, column: Optional[int], *, numerical: bool = True, **_):
        super().__init__(numerical=numerical)
        self.column: Optional[int] = column
        self.feature_index = None

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

        return self.compute_base_loss(D_yn, x0, loss_fn=loss_fn)

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype: str, *args, **kwargs):
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        # Get columns
        dtype_columns = list(
            filter(lambda col: col.dtype == dtype, dataset_config.columns)
        )
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}

        column_index = lut.get(int(unlearn_config.column), None)

        is_numerical = dtype == "numerical"
        return GradientDescent(
            column_index,
            numerical=is_numerical,
            **kwargs,
        )
