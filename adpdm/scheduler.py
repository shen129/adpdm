import torch
from torch import Tensor, Generator

from typing import Dict, Iterator, Tuple, Optional, Union
from tqdm import tqdm

from .mixin_utils import ModelMixin, register_to_config


class Scheduler(ModelMixin):
    """
        The class for the diffusion scheduler, which is used to 
    degrade the samples to the noisy observations and control the 
    iteration of the ddim inference process.

    Args:
        alpha (float, optional):
            The alpha value of the scheduler. Defaults to 0.005.
        num_iters (int, optional):
            The number of iterations of the scheduler. Defaults to 200.
    """
    config_name = "scheduler.json"

    @register_to_config
    def __init__(
        self, 
        alpha: float = 0.005, 
        num_iters: int = 200
    ) -> None:
        super(Scheduler, self).__init__()
        assert 1 / num_iters >= alpha > 0, "param: alpha must in (0, 1 / num_iters]"
        self._num_iters = num_iters

        self._len = num_iters
        for k, v in self._register(alpha, 2 / num_iters - alpha).items():
            self.register_buffer(k, v)

    def _register(self, *alphas: float) -> Dict[str, Tensor]:
        alpha_1, alpha_2 = alphas

        assert 0 < alpha_1 <= alpha_2 < 1.0
        alpha = torch.zeros(self._num_iters + 1, dtype=torch.float32)
        alpha[1:] = torch.linspace(alpha_1, alpha_2, self._num_iters)

        alpha_bar = torch.cumsum(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        return {
            "alpha":           alpha,
            "alpha_bar":       alpha_bar,
            "sqrt_alpha_bar":  sqrt_alpha_bar
        } 
    
    def get_start_t(self, sigma: float) -> int:
        """
        Get the start time of the DDIM inference scheduler.

        Args:
            sigma (float):
                The standard deviation of the noise embedded in the samples,
                which needs to be estimated by the snr.

        Returns:
            int:
                The start time of the DDIM inference scheduler. 
        """
        assert 0 <= sigma, "param: sigma must greater than 0"
        if sigma >= 1.0:
            return self._num_iters
        min_t = torch.argmin(torch.abs(self.sqrt_alpha_bar - sigma)).item()
        return min_t
    
    def rescale_samples(
        self, 
        samples: Tensor, 
        start_t: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Rescale the samples to the unit variance.

        Args:
            samples (Tensor):
                The samples to rescale.
            start_t (Optional[int], optional):
                The start time. Defaults to None.
                If None, the start time is set to the number of iterations.
                Denote that the rescale factor is calculated by the snr at the start time.

        Returns:
            Tuple[Tensor, Tensor]:
                The rescaled samples and the rescale factor.
        """
        start_t = self._num_iters if start_t is None else start_t
        freq_bins = samples.size(-1)
        rescale_factor = ((freq_bins - 1) ** 0.5) / self.sqrt_alpha_bar[start_t]
        # Avoid inplace operation
        samples = samples / rescale_factor
        return samples, rescale_factor
    
    def diffuse(
        self, 
        samples: Tensor, 
        sigma: Union[float, Tensor],
        *,
        generator: Optional[Generator] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Diffuse the samples with the given sigma.

        Args:
            samples (Tensor): 
                The clean samples to diffuse.
            sigma (Union[float, Tensor]): 
                The standard deviation of the noise to add to the samples.
            generator (Optional[Generator], optional): 
                The generator to use. Defaults to None.
        
        Returns:
            Tuple[Tensor, Tensor]:
                The noisy samples and the noise added to the samples.
        """
        # Assume that the sample is a spectrogram with shape (..., time, freq)
        *shape, freq = samples.shape

        # Get the fft bins of the sample
        n = (freq - 1) * 2

        # Sample the noise from a standard normal 
        # distribution with shape (..., freq) and transform 
        # it to the time-frequency domain with shape (..., freq//2+1)
        eps = torch.fft.fft(
            input=torch.randn(
                size=(*shape, n),
                generator=generator,
                dtype=torch.float32,
                device=samples.device
            ), 
            n=n, 
            dim=-1
        )[..., :freq]

        # Return the noisy samples and the 
        # noise added to the samples
        return samples + sigma * eps, eps


    def set_scheduler(
        self, 
        start: Optional[int], 
        end: Optional[int], 
        step: Optional[int],
        *,
        progress_bar_enabled: bool = True
    ) -> Iterator[Tuple[int, int]]:
        """
        Get the iterator of the DDIM inference scheduler.

        Args:
            start (Optional[int]): 
                The start time. Defaults to None.
                If None, the start time is set to the number of iterations.
            end (Optional[int]): 
                The end time. Defaults to None.
                If None, the end time is set to 0.
            step (Optional[int]): 
                The step size. Defaults to None.
                If None, the step size is set to 1.

        Returns:
            Iterator[Tuple[int, int]]:
                The iterator of the DDIM inference scheduler.
        """
        step = 1 if step is None else step
        assert 0 < step <= self._num_iters, f"step must be in (0, {self._num_iters}]" 
        start = self._num_iters if start is None else start
        end = 0 if end is None else end
        assert 0 <= end < start <= self._num_iters

        self._len = (start - end) // step
        if step != 1:
            self._len += 1 + int((start - end) % step != 0)

        def inner_iter(
            start: int, 
            end: int, 
            step: int
        ) -> Iterator[Tuple[int, int]]:
            if step == 1:
                for i in range(start, end, -1):
                    yield i, i - 1
            else:
                t = start
                pre_t = start - step
                while t > end:
                    if t > end + 1:
                        if pre_t > end:
                            yield t, pre_t
                        else:
                            pre_t = end + 1
                            yield t, pre_t
                    t = pre_t
                    pre_t -= step
                yield end + 1, end

        if progress_bar_enabled:
            return tqdm(
                iterable=inner_iter(start, end, step), 
                total=self._len, 
                desc="Inferencing...",
                leave=True,
                unit_scale=True
            )
        return inner_iter(start, end, step)
    