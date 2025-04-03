from .scheduler import Scheduler
from .mixin_utils import PipelineMixin, ModelMixin, register_to_config

import torch
from torch import Tensor
from typing import Optional


class Adpdm(PipelineMixin):
    """
    Additive Degradation Probabilistic Diffusion Model (ADPDM) pipeline. 
    """

    config_name = "adpdm.json"

    @register_to_config
    def __init__(
        self,
        network: ModelMixin,
        scheduler: Optional[Scheduler] = None
    ) -> None:
        if not isinstance(network, ModelMixin):
            raise TypeError("param: network must be a instance of ModelMixin")
        if scheduler is not None and not isinstance(scheduler, Scheduler):
            raise TypeError("param: scheduler must be a instance of Scheduler")
        if scheduler is None:
            scheduler = Scheduler()
        self.register_modules(network=network, scheduler=scheduler)

    def diffuse(
        self, 
        samples: Tensor, 
        sigma: float,
        *,
        seed: Optional[int] = None
    ) -> Tensor:
        """
        Diffuse the samples.

        Args:
            samples (Tensor):
                The samples to diffuse.
            sigma (float):
                The standard deviation of the noise embedded in the samples,
                which needs to be estimated by the snr.
            seed (Optional[int]):
                The seed of the random number generator. Defaults to None.

        Returns:
            Tensor:
                The noisy samples.
        """
        if self.scheduler.config.time_freq_equivalent and (samples.dtype != torch.complex64):
            raise ValueError("param: samples must be complex64 tensor.")
        elif (not self.scheduler.config.time_freq_equivalent) and (samples.dtype != torch.float32):
            raise ValueError("param: samples must be float32 tensor.")

        if isinstance(sigma, (tuple, list)):
            sigma = torch.tensor(
                data=sigma, 
                dtype=torch.float32, 
                device=samples.device
            )

        if isinstance(sigma, Tensor):
            if sigma.ndim == 1:
                if sigma.shape[0] != samples.shape[0]:
                    raise ValueError("param: sigma.shape[0] must equal to samples.shape[0].")
                sigma = sigma.view(-1, *([1] * (samples.ndim - 1)))
            elif sigma.ndim != 0:
                raise ValueError("param: sigma must be 1-d or 0-d float tensor.")
        elif not isinstance(sigma, float):
            raise ValueError("param: sigma must be float or 1-d or 0-d float tensor.") 

        generator = (
            None if seed is None 
            else torch.Generator(device=samples.device).manual_seed(seed)
        )
        return self.scheduler.diffuse(
            samples=samples, 
            sigma=sigma, 
            generator=generator
        )[0]

    @torch.no_grad()
    def denoise(
        self, 
        samples: Tensor,
        *,
        start_t: Optional[int] = None,
        end_t: Optional[int] = None,
        inference_step_size: int = 1,
        eta: float = 1.0,
        seed: Optional[int] = None,
        progressbar_enabled: bool = True
    ) -> Tensor:
        """
        Denoise the samples.

        Args:
            samples (Tensor):
                The samples to denoise.
            start_t (Optional[int]):
                The start time of the DDIM inference scheduler. Defaults to None.
                If None, the start time is set to the number of iterations.
            end_t (Optional[int]):
                The end time of the DDIM inference scheduler. Defaults to None.
                If None, the end time is set to 0.
            inference_step_size (int):
                The step size of the DDIM inference scheduler. Defaults to 1.
            eta (float):
                The eta of the DDIM inference scheduler. Defaults to 1.0.
            seed (Optional[int]):
                The seed of the random number generator. Defaults to None.
            progressbar_enabled (bool):
                Whether to show the progress bar. Defaults to True.
        """
        assert 0 <= eta <= 1.0, "param: eta must in [0, 1]"
        assert inference_step_size > 0, "param: inference_step_size must greater than 0"
        
        # Check the input dtype and dimension
        if samples.dtype not in [torch.complex64, torch.float32]:
            raise ValueError("param: samples must be complex64 or float32.")
        require_ndim = 4 if samples.dtype == torch.complex64 else 3
        ndim = samples.ndim

        if ndim == require_ndim - 2:
            samples = samples[None, None]
        elif ndim == require_ndim - 1:
            samples = samples[None]
        elif ndim != require_ndim:
            f = "L" if samples.dtype == torch.float32 else "T, F"
            raise ValueError(
                f"param: samples must be shape of {f'(N, 1, {f}) or (N, {f}) or ({f})'} "
                f"when dtype is {samples.dtype}."
            )

        self.network.eval()
        device = samples.device
        batch_size = samples.shape[0]

        # Get time step scheduler
        end_t = 0 if end_t is None else end_t
        scheduler = self.scheduler.set_scheduler(
            start=start_t, 
            end=end_t, 
            step=inference_step_size,
            progress_bar_enabled=progressbar_enabled
        )
        
        # Manual seed
        generator = (
            None if seed is None 
            else torch.Generator(device=device).manual_seed(seed)
        )

        # Rescale the samples
        samples, rescale_factor = self.scheduler.rescale_samples(samples, start_t)

        # Main Loop
        for i, j in scheduler:
            t = torch.tensor([i], device=device).repeat(batch_size)
            eps = self.network(samples, t)
            # Estimate the original sample.
            samples = samples - self.scheduler.sqrt_alpha_bar[i] * eps
            if j > end_t:
                sigma = torch.sqrt(
                    eta * 
                    self.scheduler.alpha[i] * 
                    self.scheduler.alpha_bar[j] / 
                    self.scheduler.alpha_bar[i]
                )
                # DDIM inference
                samples = samples + eps * torch.sqrt(
                    self.scheduler.alpha_bar[j] - sigma ** 2
                )
                samples, _ = self.scheduler.diffuse(
                    samples=samples, 
                    sigma=sigma / rescale_factor, 
                    generator=generator
                )

        # Restore the original scale and dimension
        samples = samples * rescale_factor
        if require_ndim - ndim == 2:
            samples = samples[0, 0]
        elif require_ndim - ndim == 1:
            samples = samples[:, 0]
        return samples