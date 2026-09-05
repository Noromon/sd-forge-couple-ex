"""
Independent (true latent) separation mode.

For an n-region canvas where each region may specify a different LoRA, this runs
n fully independent inference passes -- one per region -- each with its own
patched weights (base model + only that region's LoRAs merged). The results are
then blended in latent space according to the region masks.

Cost: n x sampling time.  Benefit: complete isolation of LoRA / prompt features
between regions (no cross-region bleed), for both DiT (Anima) and SD1/SDXL.
"""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing as P

from lib_couple.logging import logger


# filename -> raw LoRA state dict (read from disk once per generation session)
_LORA_SD_CACHE: dict[str, dict] = {}


def _lora_entry(name: str):
    """Resolve a LoRA name to its on-disk entry via the builtin networks registry."""
    import networks  # extensions-builtin/sd_forge_lora/networks.py (on sys.path at runtime)

    entry = networks.available_networks.get(name) or networks.available_network_aliases.get(name)
    if entry is None:
        raise KeyError(f"LoRA not found in registry: {name}")
    return entry


def _get_lora_state_dict(name: str) -> dict:
    """Load (and cache) the raw LoRA state dict for a network by name."""
    if name in _LORA_SD_CACHE:
        return _LORA_SD_CACHE[name]

    from backend.utils import load_torch_file

    entry = _lora_entry(name)
    sd = load_torch_file(entry.filename, safe_load=True)
    _LORA_SD_CACHE[name] = sd
    return sd


def _online_mode() -> bool:
    from backend.args import dynamic_args

    online = bool(dynamic_args.online_lora)
    if dynamic_args.ops.startswith("Mixed") or dynamic_args.ops.endswith("FP8"):
        online = False
    return online


def _apply_region_loras(sd_model, loras: list[tuple[str, float, float]]):
    """Reset the model to base weights and merge only the given LoRAs.

    Mirrors the framework's own ``networks.load_networks`` flow so that GGUF /
    nunchaku / token-merging / online-LoRA all behave consistently.

    Args:
        sd_model: the Forge diffusion engine (shared.sd_model)
        loras: list of (name, strength_unet, strength_te)

    Returns:
        (unet_patcher, clip_patcher) with only this region's LoRAs applied.
    """
    from backend.args import dynamic_args

    unet = sd_model.forge_objects_original.unet
    clip = sd_model.forge_objects_original.clip

    if dynamic_args.nunchaku:
        # nunchaku keeps active LoRAs as a (filename, strength) list on the DiT.
        unet.model.diffusion_model.loras.clear()
        for name, strength_unet, _ in loras:
            entry = _lora_entry(name)
            unet.model.diffusion_model.loras.append((entry.filename, strength_unet))
        return unet, clip

    import networks  # builtin sd_forge_lora.networks

    online_mode = _online_mode()

    for name, strength_unet, strength_te in loras:
        sd = _get_lora_state_dict(name)
        unet, clip = networks.load_lora_for_models(
            unet,
            clip,
            sd,
            strength_unet,
            strength_te,
            filename=name,
            online_mode=online_mode,
        )

    return unet, clip


def _gaussian_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over a [n, H, W] mask stack (one filter per channel)."""
    radius = int(max(1, round(sigma * 3)))
    ksize = 2 * radius + 1

    coords = torch.arange(ksize, dtype=torch.float32) - radius
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = (g / g.sum()).to(device=m.device, dtype=m.dtype)

    # Treat the n masks as the n channels of a single image and apply a depthwise
    # separable conv: kernel is [n, 1, ksize, 1] so C_out == groups == n. Padding is
    # per-dim (radius only along the filtered axis) to keep H/W unchanged.
    x = m.unsqueeze(0)  # [1, n, H, W]

    kh = g.view(1, 1, ksize, 1).expand(m.shape[0], -1, -1, -1)
    kw = g.view(1, 1, 1, ksize).expand(m.shape[0], -1, -1, -1)

    x = torch.nn.functional.conv2d(x, kh, padding=(radius, 0), groups=m.shape[0])
    x = torch.nn.functional.conv2d(x, kw, padding=(0, radius), groups=m.shape[0])

    return x.squeeze(0)  # [n, H, W]


def _blend_latents(
    results: list[torch.Tensor],
    region_masks: torch.Tensor,
    blend_mode: str,
    feather_width: float,
) -> Optional[torch.Tensor]:
    """Blend per-region latents using (optionally feathered + normalized) masks.

    Args:
        results: list of n latent tensors, each [B, C, ...] with the last two dims spatial
        region_masks: [n, H, W] pixel-space region masks (weights), one per result
        blend_mode: "Hard" or "Feather"
        feather_width: transition width in pixels (only used when blend_mode == "Feather")

    Returns:
        blended latent with the same shape as results[0], or None if no results.
    """
    if not results:
        return None

    n = len(region_masks)
    ref = results[0]
    device = ref.device

    m = region_masks.to(device=device, dtype=torch.float32)  # [n, H, W]

    if blend_mode == "Feather" and feather_width and feather_width > 0:
        sigma = max(feather_width / 2.0, 1e-3)
        m = _gaussian_blur(m, sigma)

    # Normalize so masks sum to 1 per pixel (handles overlaps & partial coverage).
    s = m.sum(dim=0, keepdim=True).clamp(min=1e-6)
    m = m / s

    # Downsample the mask stack to latent resolution.
    lh, lw = ref.shape[-2], ref.shape[-1]
    m_latent = torch.nn.functional.interpolate(
        m.unsqueeze(0), size=(lh, lw), mode="bilinear", align_corners=False
    )  # [1, n, lh, lw]
    m_latent = m_latent.squeeze(0).to(device=device, dtype=ref.dtype)  # [n, lh, lw]

    out = torch.zeros_like(ref)
    for i, r in enumerate(results):
        if i >= n:
            break
        shape = [1] * (r.dim() - 2) + [lh, lw]
        w_i = m_latent[i].view(shape)  # broadcasts over all leading dims
        out = out + r * w_i

    return out


class PrecomputedSampler:
    """Drop-in sampler stub that returns a pre-computed blended latent.

    Installed onto p.sampler so the framework's normal sampling call becomes a
    zero-cost passthrough, letting downstream logic (i2i mask blend, HR pass,
    decode) run unchanged on our independently-sampled result.
    """

    def __init__(self, result: torch.Tensor):
        self._result = result

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        return self._result

    def sample_img2img(
        self,
        p,
        x,
        noise,
        conditioning,
        unconditional_conditioning,
        steps=None,
        image_conditioning=None,
    ):
        return self._result


def _resolve_image_conditioning(p: "P", x_init: torch.Tensor) -> Optional[torch.Tensor]:
    """Return an ``image_conditioning`` tensor whose spatial size matches ``x_init``.

    Only inpaint models actually consume this (as c_concat); for every other model it is a
    harmless dummy. We reuse the pre-computed value when available and resize it if the
    resolution changed (e.g. the HR pass on an inpaint model), to avoid shape mismatches.
    """
    ic = getattr(p, "image_conditioning", None)

    if ic is not None:
        if tuple(ic.shape[-2:]) != tuple(x_init.shape[-2:]):
            ic = torch.nn.functional.interpolate(
                ic.float(), size=tuple(x_init.shape[-2:]), mode="nearest"
            ).to(ic.dtype)
        return ic

    # No pre-computed conditioning (e.g. a pure t2i object on an HR pass): fall back to the
    # txt2img-style dummy so inpaint models still get a correctly-shaped c_concat.
    if hasattr(p, "txt2img_image_conditioning"):
        try:
            return p.txt2img_image_conditioning(x_init)
        except Exception:  # noqa: BLE001
            pass

    return x_init.new_zeros((x_init.shape[0], 5, 1, 1))


def _patch_sampler_progress(sampler, region_idx: int, total_regions: int, steps_per_region: int):
    """Patch a sampler instance so WebUI progress reflects global position across all regions.

    The framework's progress formula is:
        progress = job_no/job_count + sampling_step/sampling_steps / job_count

    Without patching, each region resets sampling_steps to `steps` and sampling_step to 0,
    causing the bar to sweep 0→1 repeatedly. We override launch_sampling and callback_state
    so that:
      - state.sampling_steps = total_regions * steps_per_region (global denominator)
      - state.sampling_step starts at region_idx * steps_per_region (offset for this region)
    """
    # The framework's global State singleton lives on `modules.shared` (same object that
    # sd_samplers_common.py mutates via `from modules.shared import opts, state`).
    from modules import shared as _shared

    region_offset = region_idx * steps_per_region
    total_all_regions = total_regions * steps_per_region

    # --- Patch launch_sampling: set global denominator + initial offset before func() runs ---
    def _patched_launch_sampling(steps, func):
        from modules.sd_samplers_common import InterruptedException

        sampler.model_wrap_cfg.steps = steps
        sampler.model_wrap_cfg.total_steps = sampler.config.total_steps(steps)
        _shared.state.sampling_steps = total_all_regions  # n * p.steps (not just single-region steps)
        _shared.state.sampling_step = region_offset       # start at this region's offset
        _shared.state.preview_step = 0
        try:
            return func()
        except RecursionError:
            print("Encountered RecursionError during sampling; try to use a smaller rho value instead")
            return sampler.last_latent
        except InterruptedException:
            # BaseException subclass -- must be caught explicitly (not via `except Exception`)
            return sampler.last_latent

    sampler.launch_sampling = _patched_launch_sampling

    # --- Patch callback_state: add region offset to step index ---
    def _patched_callback_state(d):
        from modules.sd_samplers_common import InterruptedException

        step = d["i"]
        if sampler.stop_at is not None and step > sampler.stop_at:
            raise InterruptedException()
        _shared.state.sampling_step = step + region_offset  # global position, not per-region
        _shared.state.preview_step = step + 1
        _shared.total_tqdm.update()

    sampler.callback_state = _patched_callback_state


def run_independent_passes(
    p: "P",
    is_i2i: bool,
    x_init: torch.Tensor,
    noise_i2i: Optional[torch.Tensor],
    region_texts: list[str],
    negative_text: str,
    region_masks: torch.Tensor,
    loras_per_region: list[list[tuple[str, float, float]]],
    width: int,
    height: int,
    blend_mode: str = "Hard",
    feather_width: float = 0.0,
) -> Optional[torch.Tensor]:
    """Run one full independent inference pass per region and blend the latents.

    Args:
        p: the processing object (current batch context)
        is_i2i: whether this is an img2img-style pass (i2i first pass OR HR pass)
        x_init: initial noise (t2i) or init latent (i2i/HR) -- shared by every region pass
        noise_i2i: fresh noise for i2img-style passes (None for pure t2i)
        region_texts: cleaned per-region prompt texts (aligned to loras_per_region)
        negative_text: the negative prompt text
        region_masks: [n, H, W] pixel-space region masks
        loras_per_region: per-region list of (name, strength_unet, strength_te)
        width/height: target resolution for this pass (HR-aware)

    Returns:
        blended latent tensor, or None on interruption / failure.
    """
    from modules import prompt_parser, sd_samplers, shared
    from modules.prompt_parser import SdConditioning

    sd_model = p.sd_model
    steps = p.steps
    n = len(region_texts)

    if n == 0:
        return None

    # Save the full (all-LoRA) model state so we can restore it afterwards for
    # the HR pass / other extensions.
    saved_forge_objects = sd_model.forge_objects
    saved_after_lora = sd_model.forge_objects_after_applying_lora

    results: list[torch.Tensor] = []

    try:
        for i in range(n):
            if shared.state.interrupted or shared.state.skipped:
                logger.info(f"[Independent] Interrupted at region {i + 1}/{n}")
                break

            loras = loras_per_region[i] if i < len(loras_per_region) else []

            # 1. Swap in this region's patched weights (base + only its LoRAs).
            unet, clip = _apply_region_loras(sd_model, loras)
            sd_model.forge_objects.unet = unet
            sd_model.forge_objects.clip = clip
            sd_model.forge_objects_after_applying_lora = sd_model.forge_objects.shallow_copy()

            # 2. Encode this region's conditioning under its own model state so the
            #    text-encoder LoRA (if any) is isolated too.
            cond_i = prompt_parser.get_multicond_learned_conditioning(
                sd_model,
                SdConditioning([region_texts[i]], width=width, height=height),
                steps,
            )

            if p.cfg_scale == 1:
                uc_i = None
            else:
                uc_i = prompt_parser.get_learned_conditioning(
                    sd_model,
                    SdConditioning(
                        [negative_text],
                        is_negative_prompt=True,
                        width=width,
                        height=height,
                    ),
                    steps,
                )

            # 3. Bind a fresh sampler to this region's patched unet and run the full
            #    sampling loop from the shared initial latent / noise.
            sampler = sd_samplers.create_sampler(p.sampler_name, sd_model)

            # Patch progress reporting so WebUI shows global position across all n regions.
            _patch_sampler_progress(sampler, i, n, steps)

            if is_i2i:
                result_i = sampler.sample_img2img(
                    p,
                    x_init,
                    noise_i2i.clone() if noise_i2i is not None else torch.zeros_like(x_init),
                    cond_i,
                    uc_i,
                    image_conditioning=_resolve_image_conditioning(p, x_init),
                )
            else:
                result_i = sampler.sample(
                    p,
                    x_init,
                    cond_i,
                    uc_i,
                    image_conditioning=p.txt2img_image_conditioning(x_init),
                )

            results.append(result_i)
            logger.info(f"[Independent] Region {i + 1}/{n} done")
    finally:
        # Restore the full model state regardless of success / interruption.
        sd_model.forge_objects = saved_forge_objects
        sd_model.forge_objects_after_applying_lora = saved_after_lora

    if not results:
        return None

    return _blend_latents(results, region_masks, blend_mode, feather_width)
