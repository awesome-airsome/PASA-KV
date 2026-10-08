"""Patch initialization and exact rectangular RGB overlays."""

from __future__ import annotations

import torch


def initialize_patch(height: int, width: int, *, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("Patch height and width must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.rand((3, height, width), generator=generator, dtype=torch.float32).to(device).requires_grad_(True)


def _validate_patch(patch: torch.Tensor) -> torch.Tensor:
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError(f"Expected RGB patch [3,H,W], got {tuple(patch.shape)}")
    if not bool(torch.isfinite(patch).all()):
        raise ValueError("Patch contains non-finite values")
    return patch.clamp(0.0, 1.0)


def overlay_image(
    image: torch.Tensor,
    patch: torch.Tensor,
    *,
    x: int,
    y: int,
    value_range: str = "0_1",
) -> torch.Tensor:
    """Overlay an RGB patch on [B,3,H,W] or [3,H,W] without blending."""
    unbatched = image.ndim == 3
    data = image.unsqueeze(0) if unbatched else image
    if data.ndim != 4 or data.shape[1] != 3:
        raise ValueError(f"Expected image [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
    patch = _validate_patch(patch).to(device=data.device, dtype=data.dtype)
    x, y = int(x), int(y)
    h, w = patch.shape[-2:]
    if x < 0 or y < 0 or x + w > data.shape[-1] or y + h > data.shape[-2]:
        raise ValueError("Patch rectangle falls outside the image")
    if value_range == "-1_1":
        patch = patch * 2.0 - 1.0
    elif value_range != "0_1":
        raise ValueError("value_range must be '0_1' or '-1_1'")
    output = data.clone()
    output[:, :, y:y + h, x:x + w] = patch.unsqueeze(0)
    return output[0] if unbatched else output


def overlay_video(
    video: torch.Tensor,
    patch: torch.Tensor,
    *,
    x: int,
    y: int,
    frame_index: int = 0,
    value_range: str = "-1_1",
) -> torch.Tensor:
    """Overlay on [B,3,T,H,W]; training defaults to clip frame zero."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
    if not 0 <= frame_index < video.shape[2]:
        raise IndexError("frame_index is outside the clip")
    frame = video[:, :, frame_index]
    patched_frame = overlay_image(frame, patch, x=x, y=y, value_range=value_range)
    output = video.clone()
    output[:, :, frame_index] = patched_frame
    return output
