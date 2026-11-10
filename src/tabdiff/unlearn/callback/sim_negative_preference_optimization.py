import logging

import omegaconf
import torch

from typing import Callable, Optional, Union

from overrides import override

from .negative_preference_optimization import NegativePreferenceOptimization

NUMERICAL_REPR = "numerical"


class SimNegativePreferenceOptimization(NegativePreferenceOptimization):
    """
    Class implementation of Gradient-Ascent as a loss-function. The callback should be provided to a training, so to..


    Notes:
        Currently, this implementation only supports 'vertical' unlearning, i.e., feature-wise unlearning.
    """

    def __init__(
        self, column: Optional[int], *, numerical: bool = True, beta: float = 200.0, **_
    ):
        super().__init__(numerical=numerical, column=column, beta=beta)

    def init_mask_(self, D_yn: torch.FloatTensor):
        # feature_dim = D_yn.shape[-1]
        column_index = []
        if not self.column is None:
            column_index.append(self.column)

        self.feature_index = torch.tensor(
            column_index, dtype=torch.long, device=D_yn.device
        )

    def forward(
        self,
        D_yn: torch.FloatTensor,
        x0: Union[torch.FloatTensor, torch.LongTensor],
        xt: torch.FloatTensor,
        weight: torch.FloatTensor | float,
        *args,
        loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = torch.sub,
        **kwargs,
    ) -> torch.FloatTensor:
        """

        Args:
            D_yn ( ): Unlearn(ed) model prediction
            x0 (): Reference output
            xt (): Input (time pertrubed by noise schedule)
            weight (): lamba(t) weigth to sace.
            *args ():
            loss_fn (): Normal loss fn, defaults to torch.sub...
            **kwargs ():

        Returns:

        """
        # Here, we effectively compute (D_yn), which represent the (log) ratio of the policies
        return_objective = D_yn
        if self.feature_index is None:
            # Run once
            self.init_mask_(D_yn)

        return return_objective

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype, *args, **kwargs):
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
        const_beta_coeff = config.unlearn.beta

        is_numeric = dtype == "numerical"

        return SimNegativePreferenceOptimization(
            maybe_column_index,
            *args,
            numerical=is_numeric,
            beta=const_beta_coeff,
            **kwargs,
        )


class MixedNegativePreferenceOptimization(NegativePreferenceOptimization):
    """
    Adapted NPO formulation, using a 'retain' and 'forget' coefficeient `beta`.
        Thereby allowing to exert more control over the unlearning regularization from a control point of view.
    """

    def __init__(
        self,
        column: Optional[int],
        *args,
        retain_beta: float = 0.001,
        forget_beta: float = 200.0,
        **kwargs_,
    ):
        # For now; hardcode beta to None, as we have
        super().__init__(column, beta=retain_beta, **kwargs_)
        self.target_beta_coeff = forget_beta

    @override
    def get_beta(
        self,
        log_ratio: torch.FloatTensor,
        is_numeric: bool,
        *args,
        **kwargs,
    ) -> Union[float, torch.FloatTensor]:
        """
        Create beta-tensor based on forget and retain positions.
        Args:
            log_ratio ():
            is_numeric ():
            *args ():
            **kwargs ():

        Returns:
            torch.FloatTensor: Beta-scaling for samples, with retain_beta on all positions, except for forget_index,
             iff set. ofc
        """
        _, feature_size = log_ratio.shape
        beta_tensor: torch.FloatTensor = torch.full(
            (1, feature_size),
            self.beta_coeff,
            device=log_ratio.device,
            dtype=torch.float,
        )
        if isinstance(self.feature_index, torch.Tensor):
            # Only set forget on feature index to be forgotten.
            beta_tensor[:, self.feature_index] = self.target_beta_coeff

        return beta_tensor

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype, *args, **kwargs):
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        forget_beta = unlearn_config.forget_beta
        retain_beta = unlearn_config.retain_beta

        numerical = unlearn_config.dtype == NUMERICAL_REPR

        dtype_columns = list(
            filter(
                lambda col: col.dtype == unlearn_config.dtype, dataset_config.columns
            )
        )
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}

        if "column_index" in kwargs:

            overwrite_index = kwargs["column_index"]
            assert (
                "is_numerical" in kwargs
            ), "Need to provide overwrite index, in case of NOOP usage."
            numerical = overwrite_dtype = kwargs["is_numerical"]
            logging.info(
                f"Using overwritten index: {overwrite_index}. If None, the called method should support NOOP."
            )
            if overwrite_index is not None:
                column_index = lut[int(overwrite_index)]
            else:
                column_index = None
            # Clean-up after our hack
            del kwargs["column_index"]
        else:

            column_index = lut[int(unlearn_config.column)]

        return MixedNegativePreferenceOptimization(
            column_index,
            *args,
            numerical=numerical,
            forget_beta=forget_beta,
            retain_beta=retain_beta,
            **kwargs,
        )


class DtypeNegativePreferenceOptimization(NegativePreferenceOptimization):
    """
    Dtype dependent NPO, have a beta per dtype. Currently not implemented.
    """

    @override
    def get_beta(
        self,
        log_ratio: torch.FloatTensor,
        is_numeric: bool,
        *args,
        **kwargs,
    ) -> Union[float, torch.FloatTensor]:
        """
        Create beta-tensor based on dtype.
        Args:
            log_ratio ():
            is_numeric (): Allow to overwr
            *args ():
            **kwargs ():

        Returns:
            torch.FloatTensor: Beta-scaling for samples, with retain_beta on all positions, except for forget_index,
             iff set. ofc
        """
        # TODO: check training loop
        if not is_numeric:
            return self.cat_beta_
        elif self.numerical or is_numeric:
            return self.num_beta_

    def __init__(
        self,
        column: Optional[int],
        *args,
        beta: float = None,
        cat_beta: float = 0.001,
        num_beta: float = 200.0,
        **kwargs_,
    ):
        # For now; hardcode beta to None, as we have
        super().__init__(
            column,
            beta=beta,
            **kwargs_,
        )

        self.num_beta_ = num_beta
        self.cat_beta_ = cat_beta

    @classmethod
    def from_config_(self, config: omegaconf.DictConfig, dtype: str, *args, **kwargs):
        unlearn_config = config.unlearn
        dataset_config = config.dataset

        numerical = unlearn_config.dtype == NUMERICAL_REPR

        # Get columns
        dtype_columns = list(
            filter(lambda col: col.dtype == dtype, dataset_config.columns)
        )
        lut = {column.index: rel_indx for rel_indx, column in enumerate(dtype_columns)}

        if "column_index" in kwargs:

            overwrite_index = kwargs["column_index"]
            assert (
                "is_numerical" in kwargs
            ), "Need to provide overwrite index, in case of NOOP usage."
            numerical = overwrite_dtype = kwargs["is_numerical"]
            logging.info(
                f"Using overwritten index: {overwrite_index}. If None, the called method should support NOOP."
            )
            if overwrite_index is not None:
                column_index = lut[int(overwrite_index)]
            else:
                column_index = None
            # Clean-up after our hack
            del kwargs["column_index"]
        else:

            column_index = lut.get(int(unlearn_config.column), None)
        num_beta, cat_beta = unlearn_config.num_beta, unlearn_config.cat_beta
        used_beta = num_beta if numerical else cat_beta
        return DtypeNegativePreferenceOptimization(
            column_index,
            *args,
            numerical=numerical,
            beta=used_beta,
            cat_beta=cat_beta,
            num_beta=num_beta,
            **kwargs,
        )
