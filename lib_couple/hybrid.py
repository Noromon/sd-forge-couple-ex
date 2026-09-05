"""
Hybrid separation mode: global self-attention (base weights) + per-region
cross-attention / MLP (LoRA-patched weights).

Unlike Independent mode (n full passes), Hybrid runs a SINGLE sampling loop on
one unified latent.  Self-attention sees all spatial tokens simultaneously with
the base model's weights, so composition / pose / proportion stay globally
coherent from step 1 to the last step.  Cross-attention and MLP are routed per
region using each region's own LoRA-patched sub-modules (and per-region text
contexts encoded under that region's TE LoRAs), achieving full feature
isolation without any cross-region style bleed.

Cost: ~1 x steps (vs n x steps for Independent).  Per-step overhead is
n_regions x (cross-attn + mlp) instead of a single forward.

Supported architectures:
  - Anima (DiT): Block-level patch; per-region cross_attn + MLP deep copies.
  - SD1 / SDXL (UNet): attn2 replace-patch level; per-region q/k/v projection
    deep copies, full manual cross-attention with mask-blended outputs.

Boundary handling: regions are routed at the token level. "Hard" = winner-take-all
(every pixel belongs to one region); "Soft" = Gaussian-blended transition band so
adjacent regions share a smooth overlap instead of a hard seam.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing as P

from lib_couple.logging import logger


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _gaussian_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over a [n, H, W] mask stack (depthwise)."""
    radius = int(max(1, round(sigma * 3)))
    ksize = 2 * radius + 1

    coords = torch.arange(ksize, dtype=torch.float32) - radius
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = (g / g.sum()).to(device=m.device, dtype=m.dtype)

    x = m.unsqueeze(0)  # [1, n, H, W]
    kh = g.view(1, 1, ksize, 1).expand(m.shape[0], -1, -1, -1)
    kw = g.view(1, 1, 1, ksize).expand(m.shape[0], -1, -1, -1)

    x = F.conv2d(x, kh, padding=(radius, 0), groups=m.shape[0])
    x = F.conv2d(x, kw, padding=(0, radius), groups=m.shape[0])
    return x.squeeze(0)  # [n, H, W]


def compute_token_masks(
    region_masks: torch.Tensor,
    token_hw: tuple[int, int],
    boundary_mode: str = "Hard",
    soft_width: float = 0.0,
    soft_strength: float = 1.0,
) -> torch.Tensor:
    """Compute per-token region routing weights at the given spatial resolution.

    Args:
        region_masks: [n, H, W] pixel-space masks (raw weights).
        token_hw: (H_out, W_out) — target spatial size for routing.
        boundary_mode: "Hard" or "Soft".
        soft_width: transition band width in pixels (Soft mode only).
        soft_strength: 0..1 blend between hard-argmax and soft weights.

    Returns:
        [n, H_out, W_out] float tensor, normalized so each position sums to 1.
    """
    device = region_masks.device
    m = region_masks.to(device=device, dtype=torch.float32)  # [n, H, W]

    if boundary_mode == "Soft" and soft_width and soft_width > 0:
        sigma = max(soft_width / 4.0, 1e-3)
        m = _gaussian_blur(m, sigma)

    # Normalize so masks sum to 1 per pixel (handles overlaps & partial coverage).
    s = m.sum(dim=0, keepdim=True).clamp(min=1e-6)
    m = m / s

    if boundary_mode == "Soft" and soft_width and soft_width > 0:
        hard_idx = m.argmax(dim=0, keepdim=True)  # [1, H, W]
        onehot = torch.zeros_like(m).scatter_(0, hard_idx, 1.0)
        strength = min(max(soft_strength, 0.0), 1.0)
        m = strength * m + (1.0 - strength) * onehot
        s2 = m.sum(dim=0, keepdim=True).clamp(min=1e-6)
        m = m / s2

    elif boundary_mode == "Hard":
        hard_idx = m.argmax(dim=0, keepdim=True)  # [1, H, W]
        m = torch.zeros_like(m).scatter_(0, hard_idx, 1.0)

    h_tok, w_tok = token_hw
    m_tok = F.interpolate(
        m.unsqueeze(0), size=(h_tok, w_tok), mode="bilinear", align_corners=False
    ).squeeze(0)  # [n, H_tok, W_tok]

    s3 = m_tok.sum(dim=0, keepdim=True).clamp(min=1e-6)
    m_tok = m_tok / s3

    return m_tok


def _stack_region_masks(fc_args: dict, n_regions: int) -> torch.Tensor:
    """Stack fc_args mask_1..mask_n into [n, H, W]."""
    masks = torch.stack([fc_args[f"mask_{i}"] for i in range(1, n_regions + 1)], dim=0)
    if masks.dim() == 4:
        masks = masks.squeeze(1)
    return masks


# ---------------------------------------------------------------------------
# Weight baking / restore (shared nn.Module across patcher clones)
# ---------------------------------------------------------------------------


def _as_patcher(x):
    """Unwrap a CLIP wrapper to its ModelPatcher; pass through real patchers.

    forge_objects.clip is a `CLIP` object whose LoRA patches / model live on
    `.patcher` (a ModelPatcher), while forge_objects.unet IS a ModelPatcher.
    """
    return x.patcher if hasattr(x, "patcher") else x


def _bake_patcher(patcher):
    """Bake a ModelPatcher's pending LoRA patches into the shared nn.Module.

    patch_weight_to_device merges each key's LoRA diff onto the CURRENT weight and
    records the pristine base in the (shared) backup dict on first touch.
    """
    try:
        for key in list(patcher.patches.keys()):
            patcher.patch_weight_to_device(key)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Hybrid] Failed to bake patches: {e}")


def _restore_from_backup(patcher):
    """Restore the shared nn.Module's backed-up weights to pristine base.

    Mirrors ModelPatcher.unpatch_model (without clearing backup) so subsequent
    regions can still restore from the same base snapshot. This prevents LoRA
    diffs from accumulating across regions (merge is additive onto current weight).
    """
    try:
        from backend import utils as forge_utils

        model = patcher.model
        for k in list(patcher.backup.keys()):
            bk = patcher.backup[k]
            w = bk.weight
            if getattr(bk, "inplace_update", False):
                forge_utils.copy_to_param(model, k, w)
            else:
                forge_utils.set_attr(model, k, w)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Hybrid] Failed to restore base weights: {e}")


def _apply_region_loras_offline(sd_model, loras):
    """Apply only the given LoRAs (offline/baked) to fresh base patchers.

    Forces online_mode=False so every LoRA diff is merged into weight tensors, which
    makes deep-copied sub-modules carry the region's full LoRA effect. Mirrors
    independent._apply_region_loras but always offline.
    """
    from backend.args import dynamic_args
    from lib_couple.independent import _get_lora_state_dict, _lora_entry

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

    for name, strength_unet, strength_te in loras:
        sd = _get_lora_state_dict(name)
        unet, clip = networks.load_lora_for_models(
            unet,
            clip,
            sd,
            strength_unet,
            strength_te,
            filename=name,
            online_mode=False,  # force offline bake for reliable deep-copy
        )

    return unet, clip


def _is_anima_model(dit) -> bool:
    """Detect an Anima DiT (has .blocks of Block modules with cross_attn)."""
    blocks = getattr(dit, "blocks", None)
    if not blocks or len(blocks) == 0:
        return False
    first = blocks[0]
    return hasattr(first, "cross_attn") and hasattr(first, "adaln_modulation_self_attn")


def _iter_unet_attn_blocks(unet_model):
    """Yield every block containing an attn2 module, in forward-pass order."""
    for down in unet_model.down_blocks:
        for layer in down.layers:
            if hasattr(layer, "attn2"):
                yield layer
    mid = unet_model.mid_block
    for layer in getattr(mid, "layers", []):
        if hasattr(layer, "attn2"):
            yield layer
    for up in unet_model.up_blocks:
        for layer in up.layers:
            if hasattr(layer, "attn2"):
                yield layer


# ---------------------------------------------------------------------------
# Region preparation (shared by Anima & UNet paths)
# ---------------------------------------------------------------------------


def _prepare_regions(
    sd_model,
    loras_per_region: list[list[tuple[str, float, float]]],
    region_texts: list[str],
    width: int,
    height: int,
):
    """Prepare per-region LoRA-patched sub-modules and text contexts.

    For each region r:
      1. Apply only region r's LoRAs to fresh UNet/CLIP patchers (base).
      2. Bake the patches into the shared nn.Modules.
      3. Deep-copy the affected sub-modules so weights stay independent:
         Anima -> cross_attn + mlp per block; UNet -> q/k/v projections per attn2 layer.
      4. Encode region r's prompt under its own baked TE LoRAs (swap forge_objects.clip).

    Base weights are restored between regions and after all regions complete.

    Returns:
        (region_modules, region_contexts): architecture-specific module list and
        [n] context tensors ([seq_len, dim]) or None on encoding failure.
    """
    n_regions = len(loras_per_region)
    unet_orig = sd_model.forge_objects_original.unet
    dit_orig = unet_orig.model.diffusion_model
    is_anima = _is_anima_model(dit_orig)

    # Ensure the shared nn.Module starts from pristine base weights. The ACTIVE patcher
    # (forge_objects.unet) has baked all-LoRA weights and its backup dict holds true base.
    # Restoring from it guarantees we start clean before per-region baking.
    active_unet = sd_model.forge_objects.unet
    _restore_from_backup(active_unet)

    clip_orig = sd_model.forge_objects_original.clip
    if clip_orig is not None:
        active_clip = sd_model.forge_objects.clip
        _restore_from_backup(_as_patcher(active_clip))


    region_modules: list[list] = []
    region_contexts: list[Optional[torch.Tensor]] = []

    from backend import memory_management

    for r in range(n_regions):
        loras = loras_per_region[r] if r < len(loras_per_region) else []
        unet_r, clip_r = _apply_region_loras_offline(sd_model, loras)

        # --- Bake UNet patches into the shared module & deep-copy sub-modules ---
        _bake_patcher(unet_r)
        memory_management.load_model_gpu(unet_r)  # ensure ALL weights on GPU before copy
        dit_r = unet_r.model.diffusion_model

        if is_anima:
            blocks_copy: list[tuple] = []
            for b in dit_r.blocks:
                ca = copy.deepcopy(b.cross_attn)
                ml = copy.deepcopy(b.mlp)
                blocks_copy.append((ca, ml))
            region_modules.append(blocks_copy)
        else:
            qkv_copy: list[tuple] = []
            for layer in _iter_unet_attn_blocks(dit_r):
                a2 = layer.attn2
                qkv_copy.append(
                    (copy.deepcopy(a2.q), copy.deepcopy(a2.k), copy.deepcopy(a2.v))
                )
            region_modules.append(qkv_copy)

        # --- Bake CLIP patches & encode this region's text under its TE LoRAs.
        #     get_learned_conditioning uses forge_objects.clip.patcher, so swap it in. ---
        ctx = None
        if clip_orig is not None:
            _bake_patcher(_as_patcher(clip_r))

            saved_clip = sd_model.forge_objects.clip
            try:
                sd_model.forge_objects.clip = clip_r
                ctx = _encode_region_context(sd_model, region_texts[r], width, height)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"[Hybrid] Region {r + 1} context encoding failed ({e}); "
                    f"falling back to shared encoding..."
                )
                ctx = None
            finally:
                sd_model.forge_objects.clip = saved_clip

        region_contexts.append(ctx.to(dtype=torch.float32) if ctx is not None else None)

        # --- Restore base before the next region so diffs don't accumulate ---
        _restore_from_backup(unet_r)
        if clip_orig is not None:
            _restore_from_backup(_as_patcher(clip_r))


        logger.info(
            f"[Hybrid] Region {r + 1}/{n_regions} prepared "
            f"({len(loras)} LoRAs, ctx={'ok' if ctx is not None else 'fallback'})"
        )

    return region_modules, region_contexts


def _encode_region_context(sd_model, text: str, width: int, height: int) -> torch.Tensor:
    """Encode a prompt with the CURRENTLY ACTIVE CLIP weights.

    Returns a [seq_len, dim] context tensor (the same layout that mapping.text2cond
    exposes via cond_i[0][0]).
    """
    from modules.prompt_parser import SdConditioning

    texts = SdConditioning([text], False, width, height, None)
    cond = sd_model.get_learned_conditioning(texts)

    try:
        c = cond[0][0]
    except Exception:  # noqa: BLE001
        raise RuntimeError("[Hybrid] Unexpected conditioning layout...")

    if isinstance(c, dict):
        c = c.get("crossattn", None)

    if not torch.is_tensor(c):
        raise RuntimeError("[Hybrid] Unexpected conditioning tensor...")

    return c


def _build_region_contexts(
    region_contexts_raw: list[Optional[torch.Tensor]],
    fc_args: dict,
    n_regions: int,
    device,
    dtype,
) -> tuple[list[torch.Tensor], int]:
    """Normalize per-region contexts to [1, common_seq, dim] on device/dtype.

    Falls back to the framework-encoded cond_i for any region whose own encoding failed.
    Returns (padded_contexts, common_seq).
    """
    ctxs: list[torch.Tensor] = []
    for i in range(n_regions):
        c = region_contexts_raw[i]
        if c is None:
            c = fc_args[f"cond_{i + 1}"][0][0].to(dtype=torch.float32)
        ctxs.append(c.to(device=device, dtype=dtype).unsqueeze(0))

    common_seq = max(c.shape[1] for c in ctxs)
    if common_seq > 512:
        common_seq = math.ceil(common_seq / 512) * 512
    ctxs = [F.pad(c, (0, 0, 0, common_seq - c.shape[1])) for c in ctxs]
    return ctxs, common_seq


# ---------------------------------------------------------------------------
# Anima (DiT) Hybrid patching — Block level
# ---------------------------------------------------------------------------


def patch_dit_hybrid(
    sd_model,
    width: int,
    height: int,
    fc_args: dict,
    loras_per_region: list[list[tuple[str, float, float]]],
    region_texts: list[str],
    boundary_mode: str = "Hard",
    soft_width: float = 0.0,
    soft_strength: float = 1.0,
):
    """Patch the Anima DiT for Hybrid separation mode (Block-level).

    Each Block.forward is replaced so that:
      1. Self-attention runs globally with base weights (composition coherence).
      2. Cross-attention routes per region via LoRA-patched modules + per-region text.
      3. MLP routes per region via LoRA-patched modules.

    Returns the patched unet wrapper, or None on failure.
    """
    from einops import rearrange
    from modules.devices import device, dtype

    # Clone the unet wrapper and clear LoRA patches so that at sampling start
    # load_model_gpu bakes nothing and the shared DiT stays pristine base (self-attn /
    # AdaLN rely on base weights; region isolation comes from deep-copied sub-modules).
    active_unet = sd_model.forge_objects.unet
    unet = active_unet.clone()
    unet.patches = {}
    unet.online_patches = {}

    dit = unet.model.diffusion_model
    num_blocks = len(dit.blocks)
    n_regions = len(loras_per_region)

    if n_regions == 0:
        logger.error("[Hybrid/Anima] No regions to process...")
        return None

    region_modules, region_contexts_raw = _prepare_regions(
        sd_model, loras_per_region, region_texts, width, height
    )

    if len(region_modules[0]) != num_blocks:
        logger.error(
            f"[Hybrid/Anima] Block count mismatch: active={num_blocks}, "
            f"region={len(region_modules[0])}..."
        )
        return None

    region_contexts, _ = _build_region_contexts(
        region_contexts_raw, fc_args, n_regions, device, dtype
    )

    # --- Token-level routing masks at DiT spatial resolution ---
    patch_spatial = dit.patch_spatial
    h_tok = max(height // (8 * patch_spatial), 1)
    w_tok = max(width // (8 * patch_spatial), 1)

    region_masks = _stack_region_masks(fc_args, n_regions)
    token_masks = compute_token_masks(
        region_masks, (h_tok, w_tok),
        boundary_mode=boundary_mode,
        soft_width=soft_width,
        soft_strength=soft_strength,
    ).to(device=device, dtype=dtype)  # [n, H_tok, W_tok]

    logger.info(
        f"[Hybrid/Anima] Patching {num_blocks} blocks x {n_regions} regions, "
        f"tokens={h_tok}x{w_tok}, boundary={boundary_mode}"
    )

    saved_forwards = [dit.blocks[j].forward for j in range(num_blocks)]
    dit._hybrid_saved_forwards = saved_forwards

    base_dit = sd_model.forge_objects_original.unet.model.diffusion_model

    for j in range(num_blocks):
        base_blk = base_dit.blocks[j]
        reg_mods = [region_modules[r][j] for r in range(n_regions)]
        dit.blocks[j].forward = _make_anima_hybrid_forward(
            base_blk, reg_mods, n_regions, token_masks, region_contexts
        )

    return unet


# ---------------------------------------------------------------------------
# Entry point: dispatch by architecture, with forge_objects save/restore
# ---------------------------------------------------------------------------


def run_hybrid(
    p: "P",
    fc_args: dict,
    width: int,
    height: int,
    loras_per_region: list[list[tuple[str, float, float]]],
    region_texts: list[str],
    boundary_mode: str = "Hard",
    soft_width: float = 0.0,
    soft_strength: float = 1.0,
) -> bool:
    """Patch the active model for Hybrid separation and swap it into forge_objects.

    The original unet is saved on sd_model._hybrid_saved_unet so that a post-generation
    hook can restore it (undoing in-place Anima block forwards / UNet clone patches).

    Returns True on success, False on failure.
    """
    sd_model = p.sd_model
    is_anima = _is_anima_model(
        sd_model.forge_objects_original.unet.model.diffusion_model
    )

    # Clean up any previous Hybrid Anima forwards before re-patching.
    if is_anima:
        unpatch_dit_hybrid(sd_model)

    saved_unet = sd_model.forge_objects.unet

    if is_anima:
        patched = patch_dit_hybrid(
            sd_model, width, height, fc_args, loras_per_region, region_texts,
            boundary_mode=boundary_mode, soft_width=soft_width, soft_strength=soft_strength,
        )
    else:
        patched = patch_unet_hybrid(
            sd_model, width, height, fc_args, loras_per_region, region_texts,
            boundary_mode=boundary_mode, soft_width=soft_width, soft_strength=soft_strength,
        )

    if patched is None:
        return False

    # Remember the original unet so we can restore it after this generation.
    sd_model._hybrid_saved_unet = saved_unet

    sd_model.forge_objects.unet = patched
    try:
        sd_model.forge_objects_after_applying_lora = (
            sd_model.forge_objects.shallow_copy()
        )
    except Exception:  # noqa: BLE001
        pass

    logger.info(f"[Hybrid] Installed {'Anima' if is_anima else 'UNet'} hybrid unet")
    return True


def restore_hybrid(sd_model):
    """Restore the pre-Hybrid unet after a generation completes.

    Undoes in-place Anima block forwards and swaps back the original (all-LoRA) unet
    patcher so subsequent generations / other extensions see normal state.
    """
    try:
        if _is_anima_model(
            sd_model.forge_objects_original.unet.model.diffusion_model
        ):
            unpatch_dit_hybrid(sd_model)

        saved = getattr(sd_model, "_hybrid_saved_unet", None)
        if saved is not None:
            sd_model.forge_objects.unet = saved
            try:
                sd_model.forge_objects_after_applying_lora = (
                    sd_model.forge_objects.shallow_copy()
                )
            except Exception:  # noqa: BLE001
                pass
            del sd_model._hybrid_saved_unet
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Hybrid] Failed to restore unet: {e}")


def unpatch_dit_hybrid(sd_model):
    """Restore original Anima Block forwards after Hybrid mode."""
    try:
        dit = sd_model.forge_objects.unet.model.diffusion_model
    except Exception:  # noqa: BLE001
        return
    if hasattr(dit, "_hybrid_saved_forwards"):
        for j, fwd in enumerate(dit._hybrid_saved_forwards):
            dit.blocks[j].forward = fwd
        del dit._hybrid_saved_forwards


def _make_anima_hybrid_forward(base_blk, reg_mods, n_reg, t_masks, r_ctxs):
    from einops import rearrange

    @torch.inference_mode()
    def hybrid_forward(
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype

        if adaln_lora_B_T_3D is None:
            adaln_lora_B_T_3D = torch.zeros_like(emb_B_T_D)

        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        B, T, H, W, D = x_B_T_H_W_D.shape

        # --- AdaLN (base block modulation; no LoRA on these) ---
        sa_p = (base_blk.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        ca_p = (base_blk.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        mlp_p = (base_blk.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)

        def _r(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        shift_sa_r, scale_sa_r, gate_sa_r = _r(sa_p[0]), _r(sa_p[1]), _r(sa_p[2])
        shift_ca_r, scale_ca_r, gate_ca_r = _r(ca_p[0]), _r(ca_p[1]), _r(ca_p[2])
        shift_mlp_r, scale_mlp_r, gate_mlp_r = _r(mlp_p[0]), _r(mlp_p[1]), _r(mlp_p[2])

        # === 1. Self-Attention: GLOBAL with base weights (composition coherence) ===
        normed_x = base_blk._fn(
            x_B_T_H_W_D, base_blk.layer_norm_self_attn, scale_sa_r, shift_sa_r
        )
        sa_out = base_blk.self_attn(
            rearrange(normed_x.to(compute_dtype), "b t h w d -> b (t h w) d"),
            None,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        x_B_T_H_W_D = torch.addcmul(
            x_B_T_H_W_D,
            gate_sa_r.to(residual_dtype),
            rearrange(sa_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype),
        )

        cond_or_unconds = transformer_options.get("cond_or_uncond")

        # === 2. Cross-Attention: per-region routing ===
        if not cond_or_unconds or all(cu == 1 for cu in cond_or_unconds):
            normed_x = base_blk._fn(
                x_B_T_H_W_D, base_blk.layer_norm_cross_attn, scale_ca_r, shift_ca_r
            )
            ca_out = base_blk.cross_attn(
                rearrange(normed_x.to(compute_dtype), "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            )
            x_B_T_H_W_D = torch.addcmul(
                x_B_T_H_W_D,
                gate_ca_r.to(residual_dtype),
                rearrange(ca_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype),
            )
        else:
            num_chunks = len(cond_or_unconds)
            batch_size = B // num_chunks

            def _sl(t, idx):
                return t[idx * batch_size : (idx + 1) * batch_size]

            x_chunks = list(x_B_T_H_W_D.chunk(num_chunks, dim=0))
            ctx_chunks = list(crossattn_emb.chunk(num_chunks, dim=0))

            for idx, cu in enumerate(cond_or_unconds):
                if cu == 1:
                    normed_x = base_blk._fn(
                        x_chunks[idx], base_blk.layer_norm_cross_attn,
                        _sl(scale_ca_r, idx), _sl(shift_ca_r, idx),
                    )
                    ca_out = base_blk.cross_attn(
                        rearrange(normed_x.to(compute_dtype), "b t h w d -> b (t h w) d"),
                        ctx_chunks[idx],
                        rope_emb=rope_emb_L_1_1_D,
                        transformer_options=transformer_options,
                    )
                    x_chunks[idx] = torch.addcmul(
                        x_chunks[idx],
                        _sl(gate_ca_r, idx).to(residual_dtype),
                        rearrange(ca_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W).to(residual_dtype),
                    )
                else:
                    x_cond = x_chunks[idx]  # [B_c, T, H, W, D]
                    normed_x = base_blk._fn(
                        x_cond, base_blk.layer_norm_cross_attn,
                        _sl(scale_ca_r, idx), _sl(shift_ca_r, idx),
                    )
                    flat_normed = rearrange(normed_x.to(compute_dtype), "b t h w d -> b (t h w) d")

                    # Weighted sum of per-region cross-attn outputs.
                    ca_result = torch.zeros_like(flat_normed)
                    for r in range(n_reg):
                        w_r = t_masks[r]  # [H_tok, W_tok]
                        if w_r.max().item() < 0.01:
                            continue
                        ctx_r = r_ctxs[r].repeat(batch_size, 1, 1).to(dtype=compute_dtype)
                        out_r = reg_mods[r][0](
                            flat_normed, ctx_r,
                            rope_emb=rope_emb_L_1_1_D,
                            transformer_options=transformer_options,
                        )
                        w_flat = w_r.flatten().unsqueeze(0).to(dtype=compute_dtype)  # [1, S]
                        ca_result = ca_result + out_r * w_flat.unsqueeze(-1)

                    x_chunks[idx] = torch.addcmul(
                        x_cond,
                        _sl(gate_ca_r, idx).to(residual_dtype),
                        rearrange(ca_result.to(residual_dtype), "b (t h w) d -> b t h w d", t=T, h=H, w=W),
                    )

            x_B_T_H_W_D = torch.cat(x_chunks, dim=0)

        # === 3. MLP: per-region routing ===
        cond_or_unconds = transformer_options.get("cond_or_uncond")

        if not cond_or_unconds or all(cu == 1 for cu in cond_or_unconds):
            normed_x = base_blk._fn(
                x_B_T_H_W_D, base_blk.layer_norm_mlp, scale_mlp_r, shift_mlp_r
            )
            mlp_out = base_blk.mlp(normed_x.to(compute_dtype))
            x_B_T_H_W_D = torch.addcmul(
                x_B_T_H_W_D, gate_mlp_r.to(residual_dtype), mlp_out.to(residual_dtype)
            )
        else:
            num_chunks = len(cond_or_unconds)
            batch_size = B // num_chunks

            def _sl2(t, idx):
                return t[idx * batch_size : (idx + 1) * batch_size]

            x_chunks = list(x_B_T_H_W_D.chunk(num_chunks, dim=0))

            for idx, cu in enumerate(cond_or_unconds):
                if cu == 1:
                    normed_x = base_blk._fn(
                        x_chunks[idx], base_blk.layer_norm_mlp,
                        _sl2(scale_mlp_r, idx), _sl2(shift_mlp_r, idx),
                    )
                    mlp_out = base_blk.mlp(normed_x.to(compute_dtype))
                    x_chunks[idx] = torch.addcmul(
                        x_chunks[idx],
                        _sl2(gate_mlp_r, idx).to(residual_dtype),
                        mlp_out.to(residual_dtype),
                    )
                else:
                    x_cond = x_chunks[idx]  # [B_c, T, H, W, D]
                    normed_x = base_blk._fn(
                        x_cond, base_blk.layer_norm_mlp,
                        _sl2(scale_mlp_r, idx), _sl2(shift_mlp_r, idx),
                    )

                    mlp_result = torch.zeros_like(normed_x.to(compute_dtype))
                    for r in range(n_reg):
                        w_r = t_masks[r]
                        if w_r.max().item() < 0.01:
                            continue
                        out_r = reg_mods[r][1](normed_x.to(compute_dtype))
                        w_flat = w_r.flatten().view(1, T, H, W, 1).to(dtype=compute_dtype)
                        mlp_result = mlp_result + out_r * w_flat

                    x_chunks[idx] = torch.addcmul(
                        x_cond,
                        _sl2(gate_mlp_r, idx).to(residual_dtype),
                        mlp_result.to(residual_dtype),
                    )

            x_B_T_H_W_D = torch.cat(x_chunks, dim=0)

        return x_B_T_H_W_D

    return hybrid_forward


# ---------------------------------------------------------------------------
# SD1 / SDXL (UNet) Hybrid patching — attn2 replace level
# ---------------------------------------------------------------------------


def _cross_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Multi-head scaled-dot-product cross-attention on [B, S, D] tensors."""
    B, S_q, D = q.shape
    S_k = k.shape[1]
    dh = D // n_heads

    qh = q.view(B, S_q, n_heads, dh).transpose(1, 2)
    kh = k.view(B, S_k, n_heads, dh).transpose(1, 2)
    vh = v.view(B, S_k, n_heads, dh).transpose(1, 2)

    out = F.scaled_dot_product_attention(qh, kh, vh)  # [B, H, S_q, dh]
    return out.transpose(1, 2).reshape(B, S_q, D)


def patch_unet_hybrid(
    sd_model,
    width: int,
    height: int,
    fc_args: dict,
    loras_per_region: list[list[tuple[str, float, float]]],
    region_texts: list[str],
    boundary_mode: str = "Hard",
    soft_width: float = 0.0,
    soft_strength: float = 1.0,
):
    """Patch the SD1/SDXL UNet for Hybrid separation mode (attn2-level).

    Self-attention (attn1) stays global with base weights (single pass => coherent
    composition). Cross-attention is computed per region using deep-copied LoRA-patched
    q/k/v projections + per-region text contexts, and the per-region outputs are
    mask-blended at each layer.

    Returns a patched CLONE of the unet wrapper, or None on failure.
    """
    from modules.devices import device, dtype

    base_unet = sd_model.forge_objects.unet
    unet = base_unet.clone()  # patch the clone; caller swaps forge_objects.unet

    # Clear LoRA patches on the clone so that at sampling start load_model_gpu bakes
    # nothing and the shared nn.Module stays pristine base (region isolation comes from
    # the deep-copied per-region projections, not from a global bake).
    unet.patches = {}
    unet.online_patches = {}

    unet_model = unet.model.diffusion_model
    n_regions = len(loras_per_region)

    if n_regions == 0:
        logger.error("[Hybrid/UNet] No regions to process...")
        return None

    num_attn = sum(1 for _ in _iter_unet_attn_blocks(unet_model))
    if num_attn == 0:
        logger.error("[Hybrid/UNet] No cross-attention blocks found...")
        return None

    region_modules, region_contexts_raw = _prepare_regions(
        sd_model, loras_per_region, region_texts, width, height
    )

    if len(region_modules[0]) != num_attn:
        logger.error(
            f"[Hybrid/UNet] attn2 count mismatch: base={num_attn}, "
            f"region={len(region_modules[0])}..."
        )
        return None

    region_contexts, _ = _build_region_contexts(
        region_contexts_raw, fc_args, n_regions, device, dtype
    )

    # Pixel-space routing masks (resized per-layer at call time).
    region_masks = _stack_region_masks(fc_args, n_regions).to(device=device, dtype=torch.float32)

    logger.info(
        f"[Hybrid/UNet] Patching {num_attn} attn2 layers x "
        f"{n_regions} regions, boundary={boundary_mode}"
    )

    # --- Shared per-layer state. attn2_patch (store raw inputs), attn2_replace (manual
    #     attention) and the cursor advance all run once per cross-attn block in forward
    #     order, so a wrapping counter keeps them aligned with region_modules ordering. ---
    cursor = {"idx": 0}
    stored_x: list[Optional[torch.Tensor]] = [None] * num_attn
    stored_ctx: list[Optional[torch.Tensor]] = [None] * num_attn

    def attn2_patch(q, k, v, extra_options):
        # Store the raw (unprojected) spatial input and text context for this layer so
        # the replace patch can apply per-region LoRA'd projections to them.
        idx = cursor["idx"] % num_attn
        stored_x[idx] = q
        stored_ctx[idx] = k
        return q, k, v

    def attn2_replace(q, k, v, extra_options):
        # q/k/v arrive already projected by the base model. For cond regions we ignore
        # those and recompute with per-region projections on the stored raw inputs.
        try:
            idx = cursor["idx"] % num_attn
            n_heads = int(extra_options.get("n_heads", 8))

            cond_or_unconds = extra_options.get("cond_or_uncond") if isinstance(extra_options, dict) else None
            if not cond_or_unconds or all(cu == 1 for cu in cond_or_unconds):
                return _cross_attn(q, k, v, n_heads)

            num_chunks = len(cond_or_unconds)
            batch_size = q.shape[0] // num_chunks

            x_raw = stored_x[idx]
            ctx_raw = stored_ctx[idx]

            outputs: list[torch.Tensor] = []
            pos = 0
            for ci, cu in enumerate(cond_or_unconds):
                if cu == 1:
                    # Uncond: use base-projected q/k/v slice.
                    qs = q[pos : pos + batch_size]
                    ks = k[pos : pos + batch_size]
                    vs = v[pos : pos + batch_size]
                    outputs.append(_cross_attn(qs, ks, vs, n_heads))
                    pos += batch_size
                else:
                    # Cond: per-region manual cross-attention with mask blending.
                    q_c = q[pos : pos + batch_size]  # base-projected (for shape/dtype)
                    S = q_c.shape[1]
                    D = q_c.shape[-1]

                    if x_raw is None or ctx_raw is None:
                        # Defensive fallback: uniform blend of per-region outputs.
                        logger.warning("[Hybrid/UNet] Missing stored inputs; using base k/v...")
                        outputs.append(_cross_attn(q_c, k[pos : pos + batch_size], v[pos : pos + batch_size], n_heads))
                        pos += batch_size
                        continue

                    orig_shape = extra_options.get("original_shape", None)
                    if orig_shape is not None and len(orig_shape) >= 2:
                        h_tok = max(int(orig_shape[-2]), 1)
                        w_tok = max(int(orig_shape[-1]), 1)
                    else:
                        side = int(math.sqrt(S))
                        h_tok = w_tok = side

                    if h_tok * w_tok != S:
                        m_tok = None  # cannot route spatially; fall back to uniform
                    else:
                        m_tok = compute_token_masks(
                            region_masks, (h_tok, w_tok),
                            boundary_mode=boundary_mode,
                            soft_width=soft_width,
                            soft_strength=soft_strength,
                        ).to(device=q.device, dtype=q.dtype)  # [n, H, W]

                    result = torch.zeros_like(q_c)
                    for r in range(n_regions):
                        q_r = region_modules[r][idx][0](x_raw.to(dtype=q.dtype))
                        ctx_r = region_contexts[r].repeat(batch_size, 1, 1).to(
                            device=q.device, dtype=q.dtype
                        )
                        k_r = region_modules[r][idx][1](ctx_r)
                        v_r = region_modules[r][idx][2](ctx_r)
                        out_r = _cross_attn(q_r, k_r, v_r, n_heads)

                        if m_tok is None:
                            result = result + out_r / n_regions
                        else:
                            w_flat = m_tok[r].flatten().view(1, S, 1).to(dtype=q.dtype)
                            result = result + out_r * w_flat

                    outputs.append(result)
                    pos += batch_size

            return torch.cat(outputs, dim=0)
        finally:
            cursor["idx"] = (cursor["idx"] + 1) % num_attn

    unet.set_model_attn2_patch(attn2_patch)
    unet.set_model_replace_all(attn2_replace, target="attn2")

    return unet
