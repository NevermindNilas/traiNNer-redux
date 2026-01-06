from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv - much faster than regular conv"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FastResBlock(nn.Module):
    """Ultra-fast residual block with minimal operations"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Using PReLU - single param per channel, FP16 safe
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class LightBlock(nn.Module):
    """Lightweight block using depthwise separable convolutions"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels, 3)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.dw_conv(x))


class PixelShuffleUpsampler(nn.Module):
    """Efficient upsampling using pixel shuffle (ESPCN style)"""

    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pixel_shuffle(self.conv(x)))


class DIS(nn.Module):
    """
    DIS: Direct Image Supersampling

    Why "DIS"?
    - DIS-regard complexity (minimal architecture)
    - DIS-card batch norm (faster, FP16 stable)
    - DIS-patch images fast (blazing inference)
    - DIS-tilled efficiency (pure, concentrated speed)
    - DIS-tinctly simple (no attention, no transformers, just convs)

    Design principles:
    - Minimal depth (fewer layers = faster inference)
    - No batch normalization (better for inference, FP16 stable)
    - PReLU activation (FP16 safe, learnable)
    - Pixel shuffle upsampling (efficient and high quality)
    - Global residual learning (image + learned residual)
    - Mix of regular conv and depthwise separable for speed

    Args:
        in_channels: Input image channels (default: 3 for RGB)
        out_channels: Output image channels (default: 3 for RGB)
        num_features: Number of feature channels (default: 32)
        num_blocks: Number of residual blocks (default: 4)
        scale: Upscaling factor (2, 3, or 4)
        use_depthwise: Use depthwise separable convs for extra speed
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 4,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)

        # Feature extraction body
        if use_depthwise:
            self.body = nn.Sequential(
                *[LightBlock(num_features) for _ in range(num_blocks)]
            )
        else:
            self.body = nn.Sequential(
                *[FastResBlock(num_features) for _ in range(num_blocks)]
            )

        # Feature fusion
        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling - handle different scales
        if scale == 4:
            self.upsampler = nn.Sequential(
                PixelShuffleUpsampler(num_features, num_features, 2),
                PixelShuffleUpsampler(num_features, num_features, 2),
            )
        elif scale == 3:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 3)
        elif scale == 2:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 2)
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        # Final reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear upscale for global residual (fast, TensorRT friendly)
        # Using align_corners=False for ONNX compatibility
        if self.scale == 1:
            base = x
        else:
            base = functional.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            )

        # Feature extraction
        feat = self.head_act(self.head(x))

        # Body with residual
        body_out = self.body(feat)
        body_out = self.fusion(body_out) + feat

        # Upsample and reconstruct
        out = self.upsampler(body_out)
        out = self.tail(out)

        # Global residual learning
        return out + base


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def dis_balanced(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 32,
    num_blocks: int = 12,
    scale: int = 4,
    use_depthwise: bool = False,
) -> DIS:
    return DIS(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale=scale,
        use_depthwise=use_depthwise,
    )


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def dis_fast(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 32,
    num_blocks: int = 8,
    scale: int = 4,
    use_depthwise: bool = False,
) -> DIS:
    return DIS(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale=scale,
        use_depthwise=use_depthwise,
    )


class PixelUnshuffle(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out_c = c * (self.scale**2)
        out_h = h // self.scale
        out_w = w // self.scale
        x_view = x.view(b, c, out_h, self.scale, out_w, self.scale)
        return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, out_h, out_w)


@ARCH_REGISTRY.register()
class DISTemporal(nn.Module):
    """
    DISTemporal: Recurrent Video Super-Resolution variant of DIS.
    Incorporates AnimeSR-style temporal coherency with:
    - 3-frame sliding window LR input
    - Hidden state feedback
    - Previous SR frame feedback (via PixelUnshuffle)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 4,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        # Precompute input channels for the head
        # 3 frames (LR_t-1, LR_t, LR_t+1) + prev_SR_unshuffled + state
        # in_channels * 3 + in_channels * (scale^2) + num_features
        head_in_channels = in_channels * 3 + in_channels * (scale**2) + num_features

        self.unshuffle = PixelUnshuffle(scale)

        # Gating mechanism
        self.gate_conv = nn.Conv2d(in_channels * 2, num_features, 3, padding=1)

        # Feature extraction head
        # Input: 3 frames + prev_SR_unshuffled + gated_state
        self.head = nn.Conv2d(head_in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)

        # Feature extraction body (as specified by num_blocks)
        if use_depthwise:
            self.body = nn.Sequential(
                *[LightBlock(num_features) for _ in range(num_blocks)]
            )
        else:
            self.body = nn.Sequential(
                *[FastResBlock(num_features) for _ in range(num_blocks)]
            )

        # Fusion layer (mixes extracted features)
        # Outputs num_features (next state) + out_channels*(scale^2) (image residual)
        fusion_out_channels = num_features + out_channels * (scale**2)
        self.fusion = nn.Conv2d(num_features, fusion_out_channels, 3, padding=1)

        # Final reconstruction (PixelShuffle)
        self.upsampler = nn.PixelShuffle(scale)

    def forward_single(
        self, x: torch.Tensor, prev_sr: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes a single frame given history."""
        # 1. Bidirectional Motion Gating
        # x: (B, 3*C, H, W) -> [prev, cur, nxt]
        curr_lr = x[:, self.in_channels : self.in_channels * 2, :, :]
        diff_prev = torch.abs(curr_lr - x[:, : self.in_channels, :, :])
        diff_nxt = torch.abs(curr_lr - x[:, self.in_channels * 2 :, :, :])
        motion_map = (diff_prev + diff_nxt) * 0.5

        # Compute forget gate
        gate_input = torch.cat([curr_lr, motion_map], dim=1)
        forget_gate = torch.sigmoid(self.gate_conv(gate_input))

        # Apply gate to state
        gated_state = state * forget_gate

        # 2. Standard temporal processing
        # Unshuffle previous SR to LR space
        prev_sr_unshuffled = self.unshuffle(prev_sr)

        # Concatenate: [LR_window, prev_SR, gated_state]
        inp = torch.cat([x, prev_sr_unshuffled, gated_state], dim=1)

        # Feature extraction
        feat = self.head_act(self.head(inp))
        body_out = self.body(feat)

        # Fuse and split into next state and image residual
        out = self.fusion(body_out)

        # Split: the first out_channels*(scale^2) are the image residual
        res_channels = self.out_channels * (self.scale**2)
        sr_res_unshuffled = out[:, :res_channels, :, :]
        next_state = out[:, res_channels:, :, :]

        # Global residual learning (add base LR frame)
        # Extract current frame (middle of the 3-frame window)
        curr_lr = x[:, self.in_channels : self.in_channels * 2, :, :]
        base = functional.interpolate(
            curr_lr, scale_factor=self.scale, mode="bilinear", align_corners=False
        )

        out_img = self.upsampler(sr_res_unshuffled) + base

        return out_img, next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - Sequence of frames
        Returns:
            out: (B, C, H*scale, W*scale) or (B, T, C, H*scale, W*scale)
        """
        # If input is 4D (B, C, H, W), treat as single frame but this model needs temporal context
        if x.ndim == 4:
            # Pad temporal dimension to 3 frames by repeating
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)

        b, t, c, h, w = x.shape

        # Initialize state and previous output
        state = torch.zeros(
            b, self.num_features, h, w, device=x.device, dtype=x.dtype
        )
        prev_output = torch.zeros(
            b,
            self.out_channels,
            h * self.scale,
            w * self.scale,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        # Process sequence
        for i in range(t):
            # Form 3-frame window [i-1, i, i+1]
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(t - 1, i + 1)

            window = torch.cat([x[:, idx_prev], x[:, idx_curr], x[:, idx_next]], dim=1)

            prev_output, state = self.forward_single(window, prev_output, state)
            outputs.append(prev_output)

        # If training with PairedVideoDataset (which usually targets the center frame),
        # return the center frame or the last frame?
        # Mimicking TSPAN approach: return the frame corresponding to the GT.
        # PairedVideoDataset returns GT for t = clip_size // 2.
        if self.training:
            return outputs[t // 2]

        # For inference, return the whole sequence or just the last?
        # Standard video SR returns the whole sequence.
        return torch.stack(outputs, dim=1)


@ARCH_REGISTRY.register()
def distemporal_balanced(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 32,
    num_blocks: int = 12,
    scale: int = 4,
    use_depthwise: bool = False,
) -> DISTemporal:
    return DISTemporal(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale=scale,
        use_depthwise=use_depthwise,
    )


@ARCH_REGISTRY.register()
def distemporal_fast(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 32,
    num_blocks: int = 8,
    scale: int = 4,
    use_depthwise: bool = False,
) -> DISTemporal:
    return DISTemporal(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale=scale,
        use_depthwise=use_depthwise,
    )
