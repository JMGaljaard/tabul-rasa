import abc
from typing import Callable, Optional

import omegaconf
import torch

from tabdiff.auto import UnlearnCallbackRegistry


class UnlearnCallback(torch.nn.Module, metaclass=UnlearnCallbackRegistry):

    def __init__(self, numerical: bool = None, *args, **kwargs):
        super(UnlearnCallback, self).__init__()
        self.numerical = numerical

    @abc.abstractmethod
    def from_config_(
        self,
        config: omegaconf.DictConfig,
        dtype: str,
        *args,
        direction: str = "vertical",
        **kwargs,
    ):
        """
        Private constructor to build callback for unlearning.
        Args:
            config ():
            dtype ():
            *args ():
        Keyword Arguments:
            direction (str): Direction of unlearning, to differentiate between hortizonal and vertical unlearning.
            **kwargs ():

        Returns:

        """
        raise NotImplementedError("Call implemention specific method.")

    @classmethod
    def from_config(cls, config, dtype: str, *args, **kwargs):
        return cls.get_registry()[config.unlearn.method].from_config_(
            config, dtype, *args, **kwargs
        )

    @abc.abstractmethod
    def forward(
        self,
        D_yn,
        x0,
        xt,
        weight: Optional[torch.FloatTensor],
        *args,
        loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = None,
        raw_logits: torch.Tensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Method implementing the computation of the loss-component of the unlearning process.
        Args:
            D_yn (torch.FloatTensor): Tensor containing model's logits.
            x0 (): Tensor containing target variable, wei
            weight (torch.FloatTensor): Tensor containing weight(ed) score, e.g. ELBO weight, or s(t).
            loss_fn (Callable, *, None): Loss function (of the learner, to compute the base loss).
            raw_logits (torch.Tensor): Tensor containing raw logits / logsoftmaxed of the model (discrete).

        Returns:
            torch.FloatTensor: Tensor with the neg-grad applied column-wise (for now). Note


        Returns:
            torch.FloatTensor: Tensor with the unlearning objective applied, see implementation for specifics.
        """

    def compute_base_loss(self, D_yn, target, loss_fn, *args, **kwargs):
        """
        Base loss computation, to compute the base value of the lsos.
        Args:
            D_yn (torch.FloatTensor): Model predictions.
            target (torch.Tensor): Target values.
            *args (): Optional arguments
            **kwargs ():

        Returns:

        """
        # TODO: Differentiate between kwargs and args for loss-fn

        if loss_fn is not None:
            ordinary_objective = loss_fn(D_yn, target, *args, **kwargs)
        else:
            if self.numerical:
                # Ordinary grad. component, minimize

                # Support old behavior to work with MSE loss (without reduce)
                ordinary_objective = (D_yn - target) ** 2

            else:
                # TODO: right now this approach seems a bit janky. As an alternative, we could consider adding more
                #   more terms
                # Gather the logexpsums, to create softmax, selecting the the 'target' column
                ordinary_objective = torch.gather(D_yn, -1, target[:, :, None]).squeeze(
                    -1
                )
        # This returns the ** full** loss, wihtout considering only the 'masked' types
        return ordinary_objective
