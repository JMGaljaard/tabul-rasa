from functools import partial
from typing import Callable, Optional

import omegaconf
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from tabdiff.models.modules import UniModMLP
from tabdiff.models.noise_schedule import *

from tabdiff.models.unified_ctime_diffusion import (
    MixedDiffusionModelOutput,
    UnifiedDiffusionOutput,
)
from tabdiff.models.variance_schedule import LinearSchedule, VarianceSchedule
from tabdiff.unlearn.callback import UnlearnCallback

eps = 1e-8


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def index_to_log_onehot(x, num_classes):
    log_x = torch.log(x.float().clamp(min=1e-30))

    return log_x


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def to_one_hot(x_cat, num_classes):
    content = []
    for i in range(len(num_classes)):
        one_hot = F.one_hot(x_cat[:, i], num_classes=num_classes[i] + 1)
        content.append(one_hot)
    if content:
        x_cat_oh = torch.cat(content, dim=-1)

    return x_cat_oh


def get_loss_fn(loss_name: str, reduce: str = "none"):
    """
    Retrieve loss fn
    Args:
        loss_name ():

    Returns:

    """

    match loss_name:
        case "mse":
            return partial(F.mse_loss, reduction=reduce)
        case "l1":
            return partial(F.l1_loss, reduction=reduce)
        case "smooth_l1":
            return partial(F.smooth_l1_loss, reduction=reduce)
        case "huber":
            return partial(F.huber_loss, reduction=reduce)


class UnifiedDtimeDiffusion(torch.nn.Module):

    @classmethod
    def from_config(
        cls,
        config: omegaconf.DictConfig,
        y_only_model,
        *args,
        device="cuda",
        **kwargs,
    ):
        categories = (
            torch.tensor(config.unimodmlp_params.categories, device="cpu") - 1
        )  # account for missingness in data
        d_numerical = config.unimodmlp_params.d_numerical
        denoise_fn = kwargs.get("denoise_fn", None)
        kwargs["denoise_fn"] = None
        del kwargs["denoise_fn"]
        if denoise_fn is None:
            denoise_fn = UniModMLP.from_config(config.unimodmlp_params, ...)

        impute_model = kwargs.get("y_only_model", None)
        kwargs["y_only_model"] = None
        del kwargs["y_only_model"]

        loss_fn = partial(torch.nn.functional.mse_loss, reduction="none")
        if hasattr(config, "loss_fn"):
            loss_fn = get_loss_fn(config.loss_fn)

        return cls(
            num_classes=categories,
            num_numerical_features=d_numerical,
            denoise_fn=denoise_fn,
            # y_only_model=y_only_model,
            # y_only_model=y_only_model,
            num_timesteps=config.num_timesteps,
            scheduler=config.scheduler,
            cat_scheduler=config.cat_scheduler,
            gaussian_loss_type="mse",
            gaussian_parametrization="eps",
            multinomial_loss_type="vb_stochastic",
            parametrization="x0",
            noise_schedule_params=omegaconf.OmegaConf.to_container(
                config.noise_schedule_params
            ),
            loss_fn=loss_fn,
            # scheduler="cosine",
            **kwargs,
        )

    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn,
        num_timesteps=1000,
        scheduler="linear",
        cat_scheduler="log_linear",
        noise_dist="uniform",
        edm_params={},
        noise_dist_params={},
        noise_schedule_params={},
        sampler_params={},
        gaussian_loss_type="mse",
        gaussian_parametrization="eps",
        multinomial_loss_type="vb_stochastic",
        parametrization="x0",
        device=torch.device("cuda"),
        loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = partial(
            torch.nn.functional.mse_loss, reduction="none"
        ),
    ):

        super(UnifiedDtimeDiffusion, self).__init__()
        assert multinomial_loss_type in ("vb_stochastic", "vb_all")
        assert parametrization in ("x0", "direct")

        if multinomial_loss_type == "vb_all":
            print(
                "Computing the loss using the bound on _all_ timesteps."
                " This is expensive both in terms of memory and computation."
            )

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = (
            torch.from_numpy(
                np.concatenate(
                    [
                        num_classes[i].repeat(num_classes[i])
                        for i in range(len(num_classes))
                    ]
                )
            ).to(device)
            if len(num_classes) > 0
            else torch.tensor([]).to(device).int()
        )
        self.mask_index = torch.tensor(self.num_classes, dtype=torch.long).to(device)
        self.neg_infinity = -1000000.0
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        # Create masked version of classes
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes) + 1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(
                np.arange(offsets[i - 1], offsets[i])
            )

        self._denoise_fn = denoise_fn
        # Define objective function for the learner/denoiser
        self.loss_fn = loss_fn
        # self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params

        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []

        self.device = device

        self.num_schedule: VarianceSchedule = LinearSchedule(**noise_schedule_params)

        if self.cat_scheduler == "log_linear":
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == "log_linear_per_column":
            self.cat_schedule = LogLinearNoise_PerColumn(
                num_categories=len(num_classes), **noise_schedule_params
            )
        else:
            raise NotImplementedError(
                f"The noise schedule: {self.cat_scheduler} is not implemented for discrete data at DTIME "
            )

        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization

    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob ats the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)

        unormalized_prob[
            :, range(unormalized_prob.shape[1]), self.mask_index
        ] += self.neg_infinity

        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(
            unormalized_prob, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens, thereby adding zero to the loss
        unmasked_indices = xt != self.mask_index  # (bs, K)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix,
        where each sub-tensor is padded along the last dimension to match the largest
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat.
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat(
                (
                    t,
                    pad_value
                    * torch.ones(
                        *(t.shape[:-1]),
                        max_K - t.shape[-1],
                        dtype=t.dtype,
                        device=t.device,
                    ),
                ),
                dim=-1,
            )
            for t in splited
        ]
        out = torch.stack(padded_, dim=-2)
        return out

    def _absorbed_closs(
        self,
        model_output: torch.FloatTensor,
        x0: torch.LongTensor,
        xt: torch.LongTensor,
        sigma,
        dsigma,
        callback: Optional[UnlearnCallback] = None,
    ) -> UnifiedDiffusionOutput:
        """

        model_output (torch.FloatTensor): Models' estimate (potentially masekd
        """
        # Take the logit of the model, along the feature dimension,
        # following x0, which is batched over the last dimension
        # Lastly, we make a singel flattened tensro with logits

        # x0 of shape [batch_size, num_categorical_columns), of type long
        # Expresses the likelihood of
        # X0, is a long-based (index) that contains the 'index' of the right category

        if self.cat_scheduler in ["log_linear_unified", "log_linear_per_column"]:
            elbo_weight = -dsigma / torch.expm1(sigma)
        else:
            alpha = torch.exp(-sigma)
            elbo_weight = -1 / (1 - alpha)
        if callback is None:

            log_p_theta = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)

            loss = elbo_weight * log_p_theta
        else:
            loss = callback(model_output, x0, weight=elbo_weight, xt=xt)

        return UnifiedDiffusionOutput(
            loss=loss,
            target=x0,
            logit=model_output,
            weight=elbo_weight,
            xt=xt,
        )

    def q_xt(self, x, move_chance, strategy="hard"):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == "hard":
            move_indices = (
                torch.rand(x.shape, device=move_chance.device) < move_chance
            )  # [:, None]
            return self.hard_q_xt(self.mask_index, move_indices, x, self.num_classes)
        elif strategy == "soft":
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index + 1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:, 0] = 1 - move_chance[:, i]
                prob_i[:, -1] = move_chance[:, i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack(
                    (x[:, i] + slice_i[0], torch.ones_like(x[:, i]) * slice_i[-1]),
                    dim=-1,
                )
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(
                    soft_sample_i[:, 1] > soft_sample_i[:, 0],
                    self.mask_index[i],
                    x[:, i],
                )
            return xt, xt_soft

    @staticmethod
    # @torch.jit.script
    def hard_q_xt(mask_index, move_indices, x, num_classes):
        xt = torch.where(move_indices, mask_index, x)
        xt_soft = to_one_hot(xt, num_classes).to(move_indices.dtype)
        return xt, xt_soft


    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(
            self.log_1_min_cumprod_alpha, t, log_x_start.shape
        )

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

        return log_probs


    def sample_time(self, b, device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def mixed_loss(
        self,
        x,
        t: Optional[float | torch.FloatTensor] = None,
        x_0: (torch.FloatTensor, torch.FloatTensor) = None,
        num_callback: Optional[
            Callable[[torch.FloatTensor, ...], torch.FloatTensor]
        ] = None,
        cat_callback: Optional[
            Callable[[torch.FloatTensor, ...], torch.FloatTensor]
        ] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Compute loss of the to-be-optimized denoise_fn by applying noise to the image and evaluting the
        trainnig objective to the numerical and categorical columsn

        Notes:
            This function effectivley implements the `sde_lib` VPSDE logic + `construct_loss_fn` from Song et al.

            This function effectively implements the `DDIM` functionality + loss construction of the training loop from
            TabWak's tabsyn.main `training loop`.
        Args:
            x (): Batch of samples
            t (): Time step (optional for 'inversion' using ODE).
            x_0 (): Needed for some experimental design / debugging ODE.
            num_callback (): Numerical callback in case unlearning objective is passed.
            cat_callback (): Categorical callback incase unlearning objective is passed.

        Returns:

        """
        device = x.device

        b = x.shape[0]

        # Unpack samples (also for unlearning needed ;/)
        if not isinstance(x_0, torch.Tensor) and x_0 is not None:
            # TODO: DO we still need this?? ?
            assert not (
                num_callback is None and cat_callback is None
            ), "Require atleast one alternative handler when providing x_0"

            # Unpack provided handler
            x_num, x_cat = (
                x_0[:, : self.num_numerical_features],
                x_0[:, self.num_numerical_features :].long(),
            )

        else:
            x_num = x[:, : self.num_numerical_features]
            x_cat = x[:, self.num_numerical_features :].long()

        x_num_t = x_num
        log_x_cat_t = x_cat

        # Use the sigma cat from TabDiff instead of TabDDPM
        d_sigma_cat, sigma_cat, x_num_mean_t, x_num_std_t, t = self.sample_noise(
            x, t=None
        )

        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            # Implements perturbation of DDIM / DDPM style training
            x_num_t = x_num_mean_t + x_num_std_t * noise

        # Convert sigma_cat to the corresponding alpha and move_chance
        move_chance = -torch.expm1(
            -sigma_cat
        )  # torch.expm1 gives better numerical stability

        # Discrete forward diff (TabDiff)
        x_cat_t = x_cat
        is_learnable = self.cat_scheduler == "log_linear_per_column"
        strategy = "soft" if is_learnable else "hard"
        if x_cat is not None:
            x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance, strategy=strategy)
        else:
            # Default behavior set to empty tensor so we don't error out
            x_cat_t_soft = torch.empty_like(x_cat_t)
        # Predict orignal data (distribution) (with mask for
        # TODO: Fix forwad of PreCond module, s.t. we predict the right component
        model_out_num, model_out_cat = self._denoise_fn(
            x_num_t,
            x_cat_t_soft,
            t.squeeze(),
            sigma=None,
        )

        d_loss, c_loss = None, None
        if x_cat.numel() > 0:
            # TODO: Provide way to disentangle loss
            logits = self._subs_parameterization(
                model_out_cat, x_cat_t
            )  # log normalized probabilities, with the entry mask category being set to -inf

            # Logits are the predictions, x_cat the ground-truth, sigma_cat noise from cat schedule, and rate noise
            # note the
            d_loss = self._absorbed_closs(
                logits, x_cat, x_cat_t, sigma_cat, d_sigma_cat, callback=cat_callback
            )
            d_loss.xt_soft = x_cat_t_soft

        if x_num.shape[1] > 0:
            # Continuout loss (using ordinary DDPM loss)
            # TODO: Implement numberical callback
            c_loss = self._gaussian_loss(
                model_out_num, x_num, x_num_t, t, noise, callback=num_callback
            )

        # TODO:
        return MixedDiffusionModelOutput(
            categorical_output=d_loss,
            continuous_output=c_loss,
            time=t,
        )

    def sample_noise(self, x, t, x_num=None, x_cat=None):
        if not isinstance(x_num, torch.Tensor) and not isinstance(x_cat, torch.Tensor):
            x_num = x[:, : self.num_numerical_features]
            x_cat = x[:, self.num_numerical_features :].long()

            b = x.size(0)
        if isinstance(x_num, torch.Tensor):
            b = x_num.size(0)
            device = x_num.device
        if isinstance(x_cat, torch.Tensor):

            b = x_cat.size(0)
            device = x_cat.device
        if not isinstance(t, torch.Tensor) and t is None:
            t, pt = self.sample_time(b, device, "uniform")

        x_num_mean_t, x_num_std_t = x_num, x_num
        if x_num.shape[1] > 0:
            # Sample Gaussian posterior noise following DDIM / DDPM
            x_num_mean_t, x_num_std_t = self.num_schedule.total_variance(x0=x_num, t=t)


        dsigma_cat = None
        sigma_cat = None

        if x_cat.shape[1] > 0:
            # Discrete loss to fractional loss (abusing notation a bit
            t_frac = t / self.num_timesteps
            # TODO: Make that broadcast is not needed
            sigma_cat = self.cat_schedule.total_noise(t_frac[:, None]).cuda()
            dsigma_cat = self.cat_schedule.rate_noise(t_frac[:, None]).cuda()
            # log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            # log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        return dsigma_cat, sigma_cat, x_num_mean_t, x_num_std_t, t

    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None, :] * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.mask_index.device
        )

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(self.num_classes_expanded)), device=device
            )
            log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t)
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_p_sample(
                model_out_num, z_norm, t, clip_denoised=False
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    def sample_all(
        self,
        num_samples,
        batch_size,
        keep_nan_samples=False,
        ddim=True,
        num_time_steps=1000,
        **kwargs,
    ):
        b = batch_size

        all_samples = []
        num_generated = 0

        while num_generated < num_samples:
            sampled = self.sample(b)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]

            all_samples.append(sample)

            if sample.shape[0] != b:
                print(f"Sample contains: {b - sample.shape[0] } invalid records")
                # raise Exception("found nan value")
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen
