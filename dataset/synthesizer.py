import torch
from torch import Tensor
from torchaudio.functional import lfilter
from torchaudio.prototype.functional import sinc_impulse_response

import os
from typing import Optional, Tuple, Union
from random import randrange, random
from math import sqrt
from tqdm import tqdm


__all__ = [
    "LineSpectrumSynthesizer",
    "LinearFreqLineSynthesizer",
    "HyperbolicFreqLineSynthesizer",
    "RandomWalkFreqLineSynthesizer",
    "get_low_pass_filter",
    "get_fluctuation_amplitude",
    "GeneralSynthesizer"
]


def rand(low: float, high: float) -> float:
    assert low < high, "param: low must less than high."
    return random() * (high - low) + low


def get_low_pass_filter(
    cutoff_freq: float, 
    sample_rate: int, 
    device: str = "cuda",
    order: int = 13
) -> Tensor:
    """
    Generate a low-pass filter impulse response using the sinc impulse response function.

    Args:
        cutoff_freq (float): 
            The cutoff frequency of the low-pass filter.
        sample_rate (int): 
            The sample rate of the audio signal.
        device (str, optional): 
            The device on which the tensor will be stored. Defaults to "cuda".

    Returns:
        Tensor: The impulse response of the low-pass filter.
    """
    if order % 2 == 0:
        order += 1

    return sinc_impulse_response(
        cutoff=torch.tensor(cutoff_freq * 2 / sample_rate, dtype=torch.float32),
        window_size=order
    ).to(device)


def get_fluctuation_amplitude(
    amplitude: float,
    fluctuation: float,
    low_pass_filter: Tensor,
    length: int
) -> Tensor:
    """
    Generate a randomly fluctuating amplitude tensor based on the given length.

    Args:
        amplitude (float): The base amplitude value.
        fluctuation (float): The fluctuation factor.
        low_pass_filter (Tensor): The low-pass filter tensor.
        length (int): The length of the output tensor.

    Returns:
        Tensor: A tensor representing the fluctuating amplitude.
    """
    device = low_pass_filter.device
    noise = torch.randn(1, low_pass_filter.size(-1) * 2, device=device)
    a_coeffs = torch.zeros_like(low_pass_filter)
    a_coeffs[:, 0] = 1
    noise = lfilter(
        waveform=noise,
        a_coeffs=a_coeffs,
        b_coeffs=low_pass_filter,
        clamp=False,
        batching=True
    )
    mean_filter = torch.full(
        size=(1, 5),
        fill_value=0.2,
        device=device
    )
    a_coeffs = torch.zeros_like(mean_filter)
    a_coeffs[:, 0] = 1
    noise = lfilter(
        waveform=noise,
        a_coeffs=a_coeffs,
        b_coeffs=mean_filter, 
    )
    noise: Tensor = torch.nn.functional.interpolate(
        input=noise[None, None], 
        size=(1, length), 
        mode='bilinear'
    )[0, 0]

    max_value, _ = torch.max(noise.abs(), dim=-1, keepdim=True)
    noise = noise / max_value * fluctuation * amplitude + amplitude
    return noise[0]


class LineSpectrumSynthesizer(object):
    def __init__(
        self, 
        length: int,
        sample_rate: int, 
        device: str = "cuda"
    ) -> None:
        self.length = length
        self.sample_rate = sample_rate
        self.device = device

        self._lpf = get_low_pass_filter(
            cutoff_freq=5,
            sample_rate=sample_rate,
            device=device
        )
    
    def set_low_pass_filter(
        self,
        cutoff_freq: float = 5.0,
        order: int = 32
    ) -> None:
        self._lpf = get_low_pass_filter(
            cutoff_freq=cutoff_freq,
            sample_rate=self.sample_rate,
            device=self.device,
            order=order
        )

    def synthesize(
        self, 
        amplitude: float,
        length: int,
        freqs: Union[Tuple[float, float], float],
        fluctuation: Optional[float] = None,
        phase: Optional[float] = None,
        start_idx: Optional[int] = None, 
        sample: Optional[Tensor] = None
    ) -> Tensor:
        if isinstance(freqs, float):
            freq_start = freq_end = freqs
        elif isinstance(freqs, tuple | list):
            freq_start, freq_end = freqs
        else:
            raise ValueError(f"param: freqs must be float or tuple")
        assert 0 <= freq_start < self.sample_rate / 2, "param: freq_start must in [0, sample_rate / 2)"
        assert 0 <= freq_end < self.sample_rate / 2, "param: freq_end must in [0, sample_rate / 2)"

        # assert amplitude.ndim == 1, "param: amplitude must be 1d"
        if phase is None:
            phase = 2 * torch.pi * torch.rand(1)
        init_phase = phase.to(self.device)

        if sample is None:
            sample = torch.zeros(1, self.length)
        sample = sample.to(self.device)
        
        start_idx = start_idx or 0
        assert start_idx < self.length, "param: start_idx must in [0, length)"

        # fluctuate amplitude
        if fluctuation is not None and fluctuation > 0:
            amplitude = get_fluctuation_amplitude(
                amplitude=amplitude,
                fluctuation=fluctuation,
                low_pass_filter=self._lpf,
                length=length
            )
        elif fluctuation is None or fluctuation == 0:
            amplitude = torch.full(
                size=(length,),
                fill_value=amplitude,
                device=self.device
            )
        else:
            raise ValueError("param: fluctuation must be a positive float or None")

        end_idx = start_idx + amplitude.size(0)

        start = max(start_idx, 0)
        end = min(end_idx, self.length)

        if end <= start:
            raise ValueError("Out of range")
        
        phase = self.phase_rule(
            time_range=(start_idx, end_idx),
            freqs_range=freqs
        )

        length = end_idx - start_idx
        assert phase.ndim == 1 and phase.size(0) == length, f"param: freqs must be 1d and size of length == {length}"
        amplitude = amplitude.to(self.device)

        l = sample[0, start_idx:end_idx].size(-1)
   
        sample[0, start_idx:end_idx] += amplitude[:l] * torch.sin(2 * torch.pi * phase / self.sample_rate + init_phase)[:l]
        return sample

    def phase_rule(self, time_range: Tuple[int, int], freqs_range: Tuple[float, float]) -> Tensor:
        raise NotImplementedError


class LinearFreqLineSynthesizer(LineSpectrumSynthesizer):
    def phase_rule(self, time_range: Tuple[int, int], freqs_range: Union[Tuple[float, float], float]) -> Tensor:
        idx_start, idx_end = time_range
        if isinstance(freqs_range, float):
            freq_start = freq_end = freqs_range
        elif isinstance(freqs_range, tuple | list):
            freq_start, freq_end = freqs_range
        else:
            raise ValueError(f"param: freqs_range must be float or tuple")
        l = idx_end - idx_start
        t = torch.arange(0, l, device=self.device, dtype=torch.float32)
        return (freq_start + (freq_end - freq_start) / l * t / 2) * t


class HyperbolicFreqLineSynthesizer(LineSpectrumSynthesizer):
    def phase_rule(self, time_range: Tuple[int, int], freqs_range: Tuple[float, float]) -> Tensor:
        idx_start, idx_end = time_range
        freq_start, freq_end = freqs_range
        l = idx_end - idx_start
        delta = freq_end - freq_start + 1e-10
        t0 = 0.5 * l * (freq_start + freq_end) / delta 
        t = torch.arange(0, l, device=self.device, dtype=torch.float32)
        k = l * freq_start * freq_end / delta
        return k * torch.log(1 - (t -  l / 2) / t0)


class RandomWalkFreqLineSynthesizer(LineSpectrumSynthesizer):
    def __init__(
        self, 
        length: int,
        sample_rate: int, 
        resolution: float,
        segment_len_range: Optional[Tuple[int, int]] = None,
        device: str = "cuda",
        **params: float
    ) -> None:
        super(RandomWalkFreqLineSynthesizer, self).__init__(length, sample_rate, device)

        resolution = int(resolution * sample_rate)
        if segment_len_range is None:
            self._segment_len_l = resolution // 2
            self._segment_len_h = resolution * 3 // 2
        else:
            self._segment_len_l, self._segment_len_h = segment_len_range

        self.resolution = resolution

        self._J = params.get("J", 0.3)
        self._zeta = params.get("zeta", 1.0)
        self._g = params.get("g", sqrt(12))
        self._R, self._H = self.__get_r_h()

    def __get_r_h(self) -> tuple[Tensor, Tensor]:
        _R = self._g * torch.tensor(
            [
                [1 / 3, r12 := 1 / (2 * self._zeta)],
                [r12, 1 / (self._zeta ** 2)]
            ],
            device=self.device
        )
        _H = torch.tensor(
            [
                [1, self._zeta],
                [0, 1]
            ],
            device=self.device
        )
        return _R, _H

    def phase_rule(self, time_range: Tuple[int, int], freqs_range: Tuple[float, float]) -> Tensor:
        idx_start, idx_end = time_range
        freq_start, freq_end = freqs_range

        n = idx_end - idx_start
        t = torch.arange(0, n, device=self.device, dtype=torch.float32)

        idx = 0

        sampler = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0], device=self.device),
            covariance_matrix=self._R
        )

        freq = (freq_end + freq_start) / 2

        freqs = torch.zeros(n, device=self.device)
        freq_state = torch.tensor(
            [
                [int(round(freq * self.resolution / self.sample_rate, 0))],
                [self._J * torch.randint(-1, 2, size=(1,))]
            ],
            dtype=torch.float32,
            device=self.device
        )
        while n > 0:
            segment_len = randrange(self._segment_len_l, self._segment_len_h + 1)

            eta = 0.1 * sampler.sample().unsqueeze(-1)
            freq_state = self._H @ freq_state + eta
            freq_state[0] = torch.clip(
                input=freq_state[0],
                min=int(round(freq_start * self.resolution / self.sample_rate, 0)),
                max=int(round(freq_end * self.resolution / self.sample_rate, 0)) - 1
            )
            freq_state[1] = torch.clip(
                input=freq_state[1],
                min=-self._J,
                max=self._J
            )
            freqs[idx:idx+segment_len] = freq_state[0] * self.sample_rate / self.resolution

            idx = idx + segment_len
            n = n - segment_len
        
        return freqs * t


class GeneralSynthesizer(object):
    _p = {
        "cw": {
            "start_idx": [0, 1/12],
            "length": [3/4, 1]
        },
        "fm": {
            "start_idx": [0, 1/12],
            "bandwidth": [200, 300],
            "length": [1/3, 1/2]
        },
        "rw": {
            "start_idx": [0, 1/12],
            "length": [3/4, 1],
            "bandwidth": [100, 200],
        }
    }

    def __init__(
        self,
        length: int,
        sample_rate: int,
        resolution: float,
        fluctuation: float = 0.2,
        device: str = "cuda",
        **params
    ) -> None:
        self._p.update(params.get("cw", {}))
        self._p.update(params.get("fm", {}))
        self._p.update(params.get("rw", {}))

        self._cw_lfm = LinearFreqLineSynthesizer(
            length=length,
            sample_rate=sample_rate,
            device=device 
        )

        self._hfm = LinearFreqLineSynthesizer(
            length=length,
            sample_rate=sample_rate,
            device=device 
        )

        self._rw = RandomWalkFreqLineSynthesizer(
            length=length,
            sample_rate=sample_rate,
            resolution=resolution,
            device=device
        )

        self._l = length
        self._sr = sample_rate
        self._fluc = fluctuation

    def _synthesize(
        self,
        amp: float,
        max_bandwidth: float,
        freq_cw_range: Tuple[float, float],
        length_lfm_range: Tuple[int, int],
        length_hfm_range: Tuple[int, int],
        freq_rw_range: Tuple[float, float],
        num_cw: int = 1,
        num_lfm: int = 3,
        num_hfm: int = 2,
        num_rw: int = 2
    ) -> Tensor:
        sample = None
        for j in range(num_cw):
            sample = self._cw_lfm.synthesize(
                amplitude=amp,
                length=randrange(
                    start=int(self._p["cw"]["length"][0] * self._l),
                    stop=int(self._p["cw"]["length"][1] * self._l),
                ),
                fluctuation=self._fluc,
                freqs=rand(
                    low=freq_cw_range[j],
                    high=freq_cw_range[j + 1]
                ),
                start_idx=randrange(
                    start=int(self._p["cw"]["start_idx"][0] * self._l), 
                    stop=int(self._p["cw"]["start_idx"][1] * self._l)
                ),
                sample=sample
            )

        freq = rand(0 + max_bandwidth / 2, (self._sr - max_bandwidth) / 2)
        bandwidth = rand(*self._p["fm"]["bandwidth"])
        freq = (
            (freq - bandwidth / 2, freq + bandwidth / 2) 
            if random() > 0.5 
            else (freq + bandwidth / 2, freq - bandwidth / 2)
        )
        for j in range(num_lfm):
            length_range = length_lfm_range[j + 1] - length_lfm_range[j]
            sample = self._cw_lfm.synthesize(
                amplitude=amp,
                length=randrange(
                    start=int(self._p["fm"]["length"][0] * length_range),
                    stop=int(self._p["fm"]["length"][1] * length_range)
                ),
                fluctuation=self._fluc,
                freqs=freq,
                start_idx=randrange(
                    start=int(self._p["fm"]["start_idx"][0] * length_range + length_lfm_range[j]), 
                    stop=int(self._p["fm"]["start_idx"][1] * length_range + length_lfm_range[j])
                ),
                sample=sample
            )

        freq = rand(0 + max_bandwidth / 2, (self._sr - max_bandwidth) / 2)
        bandwidth = rand(*self._p["fm"]["bandwidth"])
        freq = (
            (freq - bandwidth / 2, freq + bandwidth / 2) 
            if random() > 0.5 
            else (freq + bandwidth / 2, freq - bandwidth / 2)
        )
        for j in range(num_hfm):
            length_range = length_hfm_range[j + 1] - length_hfm_range[j]
            sample = self._hfm.synthesize(
                amplitude=amp,
                length=randrange(
                    start=int(self._p["fm"]["length"][0] * length_range),
                    stop=int(self._p["fm"]["length"][1] * length_range)
                ),
                fluctuation=self._fluc,
                freqs=freq,
                start_idx=randrange(
                    start=int(self._p["fm"]["start_idx"][0] * length_range + length_hfm_range[j]), 
                    stop=int(self._p["fm"]["start_idx"][1] * length_range + length_hfm_range[j])
                ),
                sample=sample
            )

        max_bandwidth = max(*self._p["rw"]["bandwidth"])
        for j in range(num_rw):
            freq = rand(
                low=freq_rw_range[j] + max_bandwidth / 2,
                high=freq_rw_range[j + 1] - max_bandwidth / 2
            )
            sample = self._rw.synthesize(
                amplitude=amp,
                length=randrange(
                    start=int(self._p["rw"]["length"][0] * self._l),
                    stop=int(self._p["rw"]["length"][1] * self._l)
                ),
                freqs=(freq - max_bandwidth / 2, freq + max_bandwidth / 2),
                fluctuation=self._fluc,
                sample=sample
            )
        return sample[0]

    def synthesize(
        self,
        snr: float,
        save_dir: str,
        num_train_samples: int = 10000,
        num_val_samples: int = 4,
        num_cw: int = 1,
        num_lfm: int = 3,
        num_hfm: int = 2,
        num_rw: int = 2,
    ) -> None:
        save_dir = os.path.join(save_dir, f"snr_{snr}dB")
        train_dir = os.path.join(save_dir, "train")
        val_dir = os.path.join(save_dir, "val")

        if os.path.exists(train_dir):
            os.removedirs(train_dir)
        if os.path.exists(val_dir):
            os.removedirs(val_dir)
        if os.path.exists(save_dir):
            os.removedirs(save_dir)

        os.makedirs(save_dir, exist_ok=False)
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=False)
        os.makedirs(os.path.join(save_dir, "val"), exist_ok=False)

        num_lines = 2 + num_cw + num_rw

        amp = amp = sqrt(2 * (10 ** (snr / 10)) / num_lines)
        freq_cw_range = torch.linspace(
            start=0, 
            end=self._sr / 2, 
            steps=num_cw + 1
        ).tolist()
        length_lfm_range = torch.linspace(
            start=0, 
            end=self._l, 
            steps=num_lfm + 1
        ).tolist()
        length_hfm_range = torch.linspace(
            start=0, 
            end=self._l, 
            steps=num_hfm + 1
        ).tolist()
        freq_rw_range = torch.linspace(
            start=0, 
            end=self._sr / 2, 
            steps=num_rw + 1
        ).tolist()

        max_bandwidth = max(*self._p["fm"]["bandwidth"])
        data_list = {"train": [], "val": []}

        for i in tqdm(range(1, num_train_samples + 1), desc="Synthesize training data..."):
            torch.save(
                obj=self._synthesize(
                    amp=amp,
                    max_bandwidth=max_bandwidth,
                    freq_cw_range=freq_cw_range,
                    length_lfm_range=length_lfm_range, 
                    length_hfm_range=length_hfm_range,
                    freq_rw_range=freq_rw_range,
                    num_cw=num_cw,
                    num_lfm=num_lfm,
                    num_hfm=num_hfm,
                    num_rw=num_rw
                ).cpu(),
                f=os.path.join(train_dir, f"sample_{i}.pt")
            )

        for i in tqdm(range(1, num_val_samples + 1), desc="Synthesize validation data..."):
            torch.save(
                obj=self._synthesize(
                    amp=amp,
                    max_bandwidth=max_bandwidth,
                    freq_cw_range=freq_cw_range,
                    length_lfm_range=length_lfm_range, 
                    length_hfm_range=length_hfm_range,
                    freq_rw_range=freq_rw_range,
                    num_cw=num_cw,
                    num_lfm=num_lfm,
                    num_hfm=num_hfm,
                    num_rw=num_rw
                ).cpu(),
                f=os.path.join(val_dir, f"sample_{i}.pt")
            )
