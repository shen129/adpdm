# A simple version of the diffusers library on unet2d
# Reference link: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py


import torch
from torch import Tensor, nn, Size, LongTensor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    Timesteps,
    TimestepEmbedding
)
from typing import Optional, List, Tuple

from .mixin_utils import ModelMixin, register_to_config


ACTIVATIONS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU
}


def get_activation_func(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation {name} not supported")
    act = ACTIVATIONS[name]
    return act()


class AutoSizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True
    ) -> None:
        self._pad = (kernel_size - stride) % 2 == 1
        super(AutoSizeConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=(kernel_size - stride) // 2,
            bias=bias
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self._pad:
            hidden_states = nn.functional.pad(
                input=hidden_states, 
                pad=(0, 1, 0, 1),
                value=0.0
            )
        return self.conv(hidden_states)


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 5,
        num_groups: int = 8,
        non_linearity: str = "mish",
        embedding_dim: Optional[int] = None,
        embedding_norm: str = "default",
        dropout: float = 0.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True
    ) -> None:
        super(ResidualBlock2d, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm_1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels
        )

        self.conv_1 = AutoSizeConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1
        )

        if embedding_dim is not None:
            self.embedding_norm = embedding_norm
            if self.embedding_norm == "default":
                self.emb_proj = nn.Linear(
                    in_features=embedding_dim,
                    out_features=out_channels
                )
            elif self.embedding_norm == "scale_shift":
                self.emb_proj = nn.Linear(
                    in_features=embedding_dim,
                    out_features=out_channels * 2
                )
            else:
                raise ValueError(f"Unknown embedding_norm: {embedding_norm}")

        else:
            self.emb_proj = None
        
        self.norm_2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_2 = AutoSizeConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1
        )

        self.nonlinearity = get_activation_func(non_linearity)

        use_in_shortcut = in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
        if use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=conv_shortcut_bias
            )

        else:
            self.conv_shortcut = None

    def forward(self, hidden_states: Tensor, embeddings: Optional[Tensor] = None) -> Tensor:
        # Branch 1: shortcut path
        if self.conv_shortcut is None:
            shortcut = hidden_states
        else:
            shortcut = self.conv_shortcut(hidden_states)

        # Branch 2: main path
        hidden_states = self.norm_1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_1(hidden_states)

        if self.emb_proj is not None:
            if embeddings is None:
                raise ValueError("Embeddings must be provided if emb_proj is not None")
            embeddings = self.emb_proj(self.nonlinearity(embeddings))
            embeddings = embeddings[..., None, None]
            if self.embedding_norm == "default":
                hidden_states = self.norm_2(hidden_states + embeddings)
            else:
                scale, shift = torch.chunk(embeddings, 2, dim=1)
                hidden_states = self.norm_2(hidden_states)
                hidden_states = (1 + scale) * hidden_states + shift
        else:
            hidden_states = self.norm_2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv_2(hidden_states) + shortcut
        return hidden_states
    

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ) -> None:
        super(RMSNorm, self).__init__()
        self.eps = eps
        if isinstance(dim, int):
            dim = (dim,)
        self.dim = Size(dim)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states
    

class Downsample2d(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: float = 1e-5,
        elementwise_affine: bool = True,
        kernel_size: int = 3,
        scale_factor: int = 2
    ) -> None:
        super(Downsample2d, self).__init__()
        
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                normalized_shape=channels,
                elementwise_affine=elementwise_affine,
                eps=norm_eps
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                dim=channels,
                eps=norm_eps,
                elementwise_affine=elementwise_affine
            )
        elif (norm_type is None) or (norm_type == "none"):
            self.norm = None
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        
        self._pad = (kernel_size - scale_factor) % 2 == 1
        if use_conv:
            self.conv = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=(kernel_size - scale_factor) // 2
            )
        else:
            self.conv = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=(kernel_size - scale_factor) // 2
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        if self._pad:
            hidden_states = nn.functional.pad(
                input=hidden_states,
                pad=(0, 1, 0, 1),
                value=0.0
            )
        hidden_states = self.conv(hidden_states)
        return hidden_states
    

class Upsample2d(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: float = 1e-5,
        elementwise_affine: bool = True,
        kernel_size: int = 3,
        scale_factor: int = 2
    ) -> None:
        super(Upsample2d, self).__init__()
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                normalized_shape=channels,
                eps=norm_eps,
                elementwise_affine=elementwise_affine
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                dim=channels,
                eps=norm_eps,
                elementwise_affine=elementwise_affine
            )
        elif (norm_type is None) or (norm_type == "none"):
            self.norm = None
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        
        self.use_conv = use_conv
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = AutoSizeConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1
            )
        else:
            output_padding = (kernel_size - scale_factor) % 2
            self.conv = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=(kernel_size - scale_factor) // 2 + output_padding,
                output_padding=output_padding
            )
    
    def forward(self, hidden_states: Tensor, output_size: Optional[Size] = None) -> Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 

        if not self.use_conv:
            return self.conv(hidden_states, output_size=output_size)
        
        if output_size is None:
            hidden_states = nn.functional.interpolate(
                input=hidden_states,
                scale_factor=self.scale_factor,
                mode="nearest"
            )
        else:
            hidden_states = nn.functional.interpolate(
                input=hidden_states,
                size=output_size,
                mode="nearest"
            )

        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResidualDownBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        num_groups: int = 8,
        non_linearity: str = "swish",
        embedding_dim: Optional[int] = None,
        embedding_norm: str = "default",
        dropout: float = 0.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        num_layers: int = 1,
        add_downsample: bool = True,
        downsample_use_conv: bool = True,
        downsample_norm_type: Optional[str] = None,
        downsample_kernel_size: int = 3,
        downsample_scale_factor: int = 2
    ) -> None:
        assert num_layers >= 1, "num_layers should be >= 1"
        super(ResidualDownBlock2d, self).__init__()
        
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResidualBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    non_linearity=non_linearity,
                    embedding_dim=embedding_dim,
                    embedding_norm=embedding_norm,
                    dropout=dropout,
                    use_in_shortcut=use_in_shortcut,
                    conv_shortcut_bias=conv_shortcut_bias
                )
            )

        self.add_downsample = add_downsample
        if add_downsample:
            self.downsample_resnet = ResidualBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_groups=num_groups,
                non_linearity=non_linearity,
                embedding_dim=embedding_dim,
                embedding_norm=embedding_norm,
                dropout=dropout,
                use_in_shortcut=use_in_shortcut,
                conv_shortcut_bias=conv_shortcut_bias
            )
            self.downsample = Downsample2d(
                channels=out_channels,
                use_conv=downsample_use_conv,
                norm_type=downsample_norm_type,
                kernel_size=downsample_kernel_size,
                scale_factor=downsample_scale_factor
            )

    def forward(
        self, 
        hidden_states: Tensor, 
        embeddings: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        output_states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, embeddings)
            output_states.append(hidden_states)
        if self.add_downsample:
            hidden_states = self.downsample_resnet(hidden_states, embeddings)
            hidden_states = self.downsample(hidden_states)
            output_states.append(hidden_states)
        return hidden_states, output_states
    

class ResidualUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        num_groups: int = 8,
        non_linearity: str = "swish",
        embedding_dim: Optional[int] = None,
        embedding_norm: str = "default",
        dropout: float = 0.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        num_layers: int = 1,
        add_upsample: bool = True,
        upsample_use_conv: bool = True,
        upsample_norm_type: Optional[str] = None,
        upsample_kernel_size: int = 3,
        upsample_scale_factor: int = 2
    ) -> None:
        super(ResidualUpBlock2d, self).__init__()

        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = skip_channels if i == 0 else out_channels

            self.resnets.append(
                ResidualBlock2d(
                    in_channels=res_skip_channels + resnet_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    non_linearity=non_linearity,
                    embedding_dim=embedding_dim,
                    embedding_norm=embedding_norm,
                    dropout=dropout,
                    use_in_shortcut=use_in_shortcut,
                    conv_shortcut_bias=conv_shortcut_bias
                )
            )
        
        self.add_upsample = add_upsample
        if add_upsample:
            self.upsample_resnet = ResidualBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_groups=num_groups,
                non_linearity=non_linearity,
                embedding_dim=embedding_dim,
                embedding_norm=embedding_norm,
                dropout=dropout,
                use_in_shortcut=use_in_shortcut,
                conv_shortcut_bias=conv_shortcut_bias
            )
            self.upsample = Upsample2d(
                channels=out_channels,
                use_conv=upsample_use_conv,
                norm_type=upsample_norm_type,
                kernel_size=upsample_kernel_size,
                scale_factor=upsample_scale_factor
            )

    def forward(
        self, 
        hidden_states: Tensor, 
        skip_hidden_states_list: List[Tensor],
        embeddings: Optional[Tensor] = None,
        output_size: Optional[Size] = None
    ) -> Tensor:
        for resnet in self.resnets:
            skip_hidden_states = skip_hidden_states_list.pop()
            hidden_states = torch.cat((hidden_states, skip_hidden_states), dim=1)
            hidden_states = resnet(hidden_states, embeddings)
        if self.add_upsample:
            hidden_states = self.upsample_resnet(hidden_states, embeddings)
            hidden_states = self.upsample(hidden_states, output_size)
        return hidden_states
    

class UNet2d(nn.Module):
    config_name = "unet.json"

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        block_out_channels: Tuple[int, ...] = (32, 64, 128, 256),
        kernel_size: int = 3,
        norm_num_groups: int = 8,
        non_linearity: str = "swish",
        embedding_type: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_norm: Optional[str] = None,
        layers_per_block: int = 2,
        downsample_use_conv: bool = True,
        upsample_use_conv: bool = True,
        resample_kernel_size: Optional[int] = None,
        resample_scale_factor: int = 2,
        dropout: float = 0.0
    ) -> None:
        super(UNet2d, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        resample_kernel_size = kernel_size if resample_kernel_size is None else resample_kernel_size

        # Embedding
        if (embedding_type is None) ^ (embedding_dim is None):
            raise ValueError("embedding_type and embedding_dim must be specified together") 
        embedding_norm = (
            None if embedding_dim is None 
            else ("default" if embedding_norm is None else embedding_norm)
        )
        self.embedding_type = embedding_type
        if embedding_type is not None:
            if embedding_type == "fourier":
                self.time_proj = GaussianFourierProjection(
                    embedding_size=block_out_channels[0], 
                    scale=16
                )
                timestep_input_dim = 2 * block_out_channels[0]
            elif embedding_type == "positional":
                self.time_proj = Timesteps(block_out_channels[0], True, 0)
                timestep_input_dim = block_out_channels[0]
            else:
                raise ValueError(f"Unknown embedding type: {embedding_type}")
            self.embedding = TimestepEmbedding(timestep_input_dim, embedding_dim)

        # In
        self.conv_in = AutoSizeConv2d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=kernel_size,
            stride=1
        )

        # Down
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, j in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = j
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                ResidualDownBlock2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    num_groups=norm_num_groups,
                    non_linearity=non_linearity,
                    embedding_dim=embedding_dim,
                    embedding_norm=embedding_norm,
                    dropout=dropout,
                    use_in_shortcut=None,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    downsample_use_conv=downsample_use_conv,
                    downsample_kernel_size=resample_kernel_size,
                    downsample_scale_factor=resample_scale_factor
                )
            )

        # Mid
        self.mid_block = ResidualBlock2d(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            kernel_size=kernel_size,
            num_groups=norm_num_groups,
            non_linearity=non_linearity,
            embedding_dim=embedding_dim,
            embedding_norm=embedding_norm,
            dropout=dropout,
            use_in_shortcut=None,
            conv_shortcut_bias=True
        )
        
        # Up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, j in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = j
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1

            self.up_blocks.append(
                ResidualUpBlock2d(
                    in_channels=input_channel,
                    skip_channels=prev_output_channel,
                    out_channels=output_channel,
                    kernel_size=kernel_size,
                    num_groups=norm_num_groups,
                    non_linearity=non_linearity,
                    embedding_dim=embedding_dim,
                    embedding_norm=embedding_norm,
                    dropout=dropout,
                    use_in_shortcut=None,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final_block,
                    upsample_use_conv=upsample_use_conv,
                    upsample_kernel_size=resample_kernel_size,
                    upsample_scale_factor=resample_scale_factor
                )
            )
            prev_output_channel = output_channel

        # Out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=num_groups_out
        )
        self.conv_act = nn.SiLU()
        self.conv_out = AutoSizeConv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1
        )

    def forward(self, samples: Tensor, embeddings: Optional[LongTensor] = None) -> Tensor:
        # Embedding
        if self.embedding_type is not None:
            if embeddings is None:
                raise ValueError("Embeddings must be provided if embedding_type is not None")
            assert embeddings.ndim == 1, "Embeddings must be 1D"
            embeddings = self.time_proj(embeddings)
            embeddings = self.embedding(embeddings)

        # Preprocess
        samples = self.conv_in(samples)

        skip_samples = [samples]
        output_sizes = [None]

        # Downsample path
        for down_block in self.down_blocks:
            output_sizes.append(samples.shape[-2:])
            samples, res_sample = down_block(
                hidden_states=samples, 
                embeddings=embeddings
            )
            skip_samples.extend(res_sample)
        
        # Mid path
        samples = self.mid_block(samples, embeddings)

        # Up path
        output_sizes.pop()
        for up_block in self.up_blocks:
            output_size = output_sizes.pop()
            samples = up_block(
                hidden_states=samples, 
                skip_hidden_states_list=skip_samples,
                embeddings=embeddings,
                output_size=output_size
            )
            
        # Postprocess
        samples = self.conv_norm_out(samples)
        samples = self.conv_act(samples)
        samples = self.conv_out(samples)
        return samples


class CplxUNet2d(UNet2d, ModelMixin):
    config_name = "cplx_unet.json"

    @register_to_config
    def __init__(
        self,
        block_out_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
        kernel_size: int = 3,
        norm_num_groups: int = 8,
        non_linearity: str = "swish",
        embedding_type: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_norm: Optional[str] = None,
        layers_per_block: int = 2,
        downsample_use_conv: bool = True,
        upsample_use_conv: bool = True,
        resample_kernel_size: Optional[int] = None,
        resample_scale_factor: int = 2,
        dropout: float = 0.0
    ) -> None:
        super(CplxUNet2d, self).__init__(
            in_channels=2,
            out_channels=None, 
            block_out_channels=block_out_channels,
            kernel_size=kernel_size,
            norm_num_groups=norm_num_groups,
            non_linearity=non_linearity,
            embedding_type=embedding_type,
            embedding_dim=embedding_dim,
            embedding_norm=embedding_norm,
            layers_per_block=layers_per_block,
            downsample_use_conv=downsample_use_conv,
            upsample_use_conv=upsample_use_conv,
            resample_kernel_size=resample_kernel_size,
            resample_scale_factor=resample_scale_factor,
            dropout=dropout
        )

    def forward(self, hidden_states: Tensor, embeddings: Optional[LongTensor] = None) -> Tensor:
        assert hidden_states.dtype == torch.complex64, "Input must be complex64 tensor."
        assert hidden_states.ndim == 4 and hidden_states.shape[1] == 1, "Input must be 2d-1c tensor."

        hidden_states = torch.cat((hidden_states.real, hidden_states.imag), dim=1)
        hidden_states = super(CplxUNet2d, self).forward(hidden_states, embeddings)
        return torch.complex(
            real=hidden_states[:, 0:1],
            imag=hidden_states[:, 1:2]
        )
