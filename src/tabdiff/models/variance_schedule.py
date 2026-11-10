import abc
import logging
import math

import numpy as np
import torch.nn


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


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    logging.info("Falling back to default DDPM schedules (linear / ... )")

    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    # TODO: Define schedule for conversion from TabSyn like noise to DDIM / DDPM style samplin
    #   using alpha_t = 1 / (1 + sigma_t ** 2)


class VarianceSchedule(abc.ABC):
    """
    Base class implement the interface for a Descretized Diffusion variance schedule.
    """

    def __init__(self, num_timesteps: int = 1000):
        super(VarianceSchedule, self).__init__()
        self.num_timesteps = num_timesteps

    @abc.abstractmethod
    def total_variance(self, x0, t):
        """
        Implements total_variance computation for a given variance schedule.
        Args:
            xt ():
            t ():

        Returns:

        """
        pass


class LinearSchedule(VarianceSchedule, torch.nn.Module):

    def get_beta_schedule(self):

        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / self.num_timesteps
        beta_start = scale * self.beta_min
        beta_end = scale * self.beta_max
        return np.linspace(beta_start, beta_end, self.num_timesteps, dtype=np.float64)

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_min=0.001,
        beta_max=0.02,
        clip_sample=True,
        fix_alpha=True,
        device="cuda",
    ):
        super(LinearSchedule, self).__init__(num_timesteps)
        self.device = device
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.clip_sample = clip_sample
        self.fix_alpha = fix_alpha

        # Compute Complement of variance schedule beta_t
        alphas = 1.0 - self.get_beta_schedule()
        alphas = torch.tensor(alphas.astype(np.float64))  # alpha2_t
        # Compute high-precision beta2_t
        betas = 1.0 - alphas  # beta2_t

        # Compute bar_alpha_t using log-trick (Eqn. 3--4) DDPM
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        # Compute more stable version
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # Compute alpha_bar
        alphas_cumprod = np.cumprod(alphas, axis=0)  # tilde_alpha2_t
        # Set alpha_{t-1} to 1 for t=0 (doen't exist otherwise)
        alphas_cumprod_prev = torch.tensor(
            np.append(1.0, alphas_cumprod[:-1])
        )  # tilde_alpha2_{t-1}

        # Skip the first entry as this is fixed.
        alphas_cumprod_next = torch.tensor(
            np.append(alphas_cumprod[1:], 0.0)
        )  # tilde_alpha2_{t+1}

        # Compute variance
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # tilde_alpha_t

        # Compute variance
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)  # tilde_beta_t
        sqrt_recip_alphas_cumprod = np.sqrt(
            1.0 / alphas_cumprod
        )  # sqrt(1 / tilde_alpha_t)
        sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / alphas_cumprod - 1
        )  # sqrt(tilde_beta_t / tilde_alpha_t )

        # Gaussian diffusion
        # Posterior variance schedule Eqn. 7 DDPM page 3.
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Compute log-transformed version, prevent raising exception of nan output on indx. 0
        self.posterior_log_variance_clipped = (
            torch.from_numpy(
                np.log(
                    np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
            )
            .float()
            .to(self.device)
        )

        # Posterior (x_0 * factor)
        self.posterior_mean_coef1 = (
            (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
            .float()
            .to(self.device)
        )
        # ...
        self.posterior_mean_coef2 = (
            (
                (1.0 - alphas_cumprod_prev)
                * np.sqrt(alphas.numpy())
                / (1.0 - alphas_cumprod)
            )
            .float()
            .to(self.device)
        )

        # check for numerical precision in computation.
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item()
            < 1e-5
        )
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.0e-5

        # Convert to float32 and register buffers.
        self.register_buffer("alphas", alphas.float().to(device))
        self.register_buffer("log_alpha", log_alpha.float().to(device))
        self.register_buffer("log_1_min_alpha", log_1_min_alpha.float().to(device))
        self.register_buffer(
            "log_1_min_cumprod_alpha", log_1_min_cumprod_alpha.float().to(device)
        )
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha.float().to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.float().to(device))
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.float().to(device)
        )
        self.register_buffer(
            "alphas_cumprod_next", alphas_cumprod_next.float().to(device)
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", sqrt_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod.float().to(device),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            sqrt_recipm1_alphas_cumprod.float().to(device),
        )

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def total_variance(self, x0, t):
        """
        Compute marginal distribution mean and std given a sample and timestep t. See also
        self.gaussian_q_sample
        Args:
            x0 (): Sample at x0
            t (): Timestep to construct marginal distribution statistics for

        Returns:

        """
        mean = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        std = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return mean, std

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self,
        model_output,
        model_pred_x0,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=model_pred_x0, x_t=x, t=t
        )
        model_variance = torch.cat(
            [
                self.posterior_variance[1].unsqueeze(0).to(x.device),
                (1.0 - self.alphas)[1:],
            ],
            dim=0,
        )
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        assert (
            model_mean.shape
            == model_log_variance.shape
            == model_pred_x0.shape
            == x.shape
        ), f"{model_mean.shape}, {model_log_variance.shape}, {model_pred_x0.shape}, {x.shape}"

        return model_mean, model_variance, model_log_variance
