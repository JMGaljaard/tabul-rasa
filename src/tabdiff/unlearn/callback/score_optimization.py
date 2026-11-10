import numpy as np
import omegaconf
import torch

from typing import Callable, Optional, Union
from .template import UnlearnCallback

import torch.nn.functional as F

NUMERICAL_REPR = "numerical"

# TODO: Maybe rewrite as a mixin, so we can combine different aspects (like common reguarlizers)


class ScoreOrthogonal(UnlearnCallback):
    """
    Class implementation of Orthogonal Score optimization.


    Notes:
        Currently, this implementation only supports 'vertical' unlearning, i.e., feature-wise unlearning.
    """

    def __init__(
        self,
        column: Optional[int],
        *,
        numerical: bool = True,
        categories: Optional[np.ndarray],
        **_,
    ):
        super().__init__(numerical=numerical)
        self.column: Optional[int] = column
        self.feature_index = None
        self.categories = None
        if categories is not None:
            self.categories = torch.from_numpy(categories)

    def init_mask_(self, D_yn: torch.FloatTensor):
        # feature_dim = D_yn.shape[-1]
        self.feature_index = torch.tensor(
            [self.column], dtype=torch.long, device=D_yn.device
        )
        if (
            isinstance(self.categories, torch.Tensor)
            and D_yn.device != self.categories.device
        ):
            self.categories = self.categories.to(D_yn.device)

    def forward(
        self,
        D_yn: torch.FloatTensor,
        x0: Union[torch.FloatTensor, torch.LongTensor],
        *args,
        weight: torch.FloatTensor = None,
        xt: Union[torch.FloatTensor, torch.LongTensor] = None,
        loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = None,
        raw_logits: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Method implementing the computation of the loss-component of the unlearning process.
        Args:
            D_yn (torch.FloatTensor): Tensor containing model's logits.
            x0 (): Tensor containing target variable, wei
            weight: Re-weighting for the objective.
            loss_fn (Callable): Implementation of the base-loss function.
            D_yn (torch.FloatTensor): Tensor containing raw logits
        Returns:
            torch.FloatTensor: Tensor with the neg-grad applied column-wise (for now). Note

        """
        ordinary_objective = self.compute_base_loss(D_yn, x0, loss_fn=loss_fn)
        if self.column is None:
            # In case the column is a None type, we return ordinar objective...
            return ordinary_objective

        if self.feature_index is None:
            self.init_mask_(D_yn)

        if self.numerical:
            batch_slice = torch.arange(ordinary_objective.size(0))
            # Perform orthogonal objective
            # Note: x0 is orignal sample, D_yn is x_0_hat.
            # TODO: compute score and noise vector. Implement like that.
            ordinary_objective[batch_slice, self.feature_index] = (
                D_yn[batch_slice, self.feature_index]
                * x0[batch_slice, self.feature_index]
            ) ** 2
        else:
            # TODO:
            # Create for each output a distributoin
            # THeoretically can can mask the output using arange, and knowledge of the number of classes
            # masky_mask = (
            #     torch.arange(0, D_yn.size(2)).unsqueeze(0).repeat((D_yn.size(1), 1))
            # )
            # TODO: Retrieve the number of lcasses, either Dynamically
            reference_distribution = torch.zeros_like(D_yn[0], dtype=torch.float)

            for i in range(reference_distribution.shape[0]):
                reference_distribution[i, : self.categories[i]] = (
                    1.0 / self.categories[i]
                )
                reference_distribution[i, self.categories[i] :] = 0.0
            # TODO: As not every vector has the same 'distribution' solve this through either
            #   Loop over the number of categories, and averaging their results manually...
            #   Alternative: Use masky / mask / arange to create correct lut tensors to compute
            # Performs sample mean reduction, instead of batch-mean
            div_weight = F.kl_div(
                D_yn[:, self.column, : self.categories[self.feature_index]],
                reference_distribution[
                    self.feature_index, : self.categories[self.feature_index]
                ],
                reduce=False,
            ).mean(-1)
            # (unmasked positions) [ for column of interest ] ... flattened, of lenght [batch_size,]
            masked_positions = (xt == self.categories)[:, self.feature_index].squeeze(1)

            # TODO: now we need to apply these results in the indices that are unmasked, as those are not optimized anymore.
            # Set the weight of the column to the computed KL divergence with a uniform distribution.
            # In theory...
            # Now we overwrite the objective for the target with... the reduced KL divergence.
            ordinary_objective[masked_positions, self.feature_index] = div_weight[
                masked_positions
            ]
        return ordinary_objective

    @classmethod
    def from_config_(
        cls,
        config: omegaconf.DictConfig,
        dtype: str,
        *args,
        categories: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "ScoreOrthogonal":
        """

        Args:
            dtype ():
            config (omegaconf.DictConfig): Experimental configuration.
            *args ():
            **kwargs ():

        Returns:

        """
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        dtype_columns = list(
            filter(lambda col: col.dtype == dtype, dataset_config.columns)
        )
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}
        column_index = lut.get(int(unlearn_config.column), None)

        is_numeric = dtype == "numerical"
        return ScoreOrthogonal(
            column=column_index,
            numerical=is_numeric,
            categories=categories,
        )
