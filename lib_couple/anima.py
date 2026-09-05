import math
from functools import wraps

import torch

from backend.nn.anima import SelfCrossAttention
from lib_couple.logging import logger
from modules.devices import device, dtype

from .attention_masks import get_dit_mask, lcm_for_list, sharpen_mask


class AttentionCoupleAnima:

    @staticmethod
    @torch.inference_mode()
    def patch_dit(
        model,
        base_mask,
        width: int,
        height: int,
        kwargs: dict,
        separation_mode: str = "Attention",
        mask_mode: str = "Soft",
        mask_temperature: float = 1.0,
    ):
        dit = model.model.diffusion_model

        num_conds = len(kwargs) // 2 + 1

        mask = [base_mask] + [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
        mask = torch.stack(mask, dim=0).to(device=device, dtype=dtype)

        if mask.sum(dim=0).min().item() <= 0.0:
            logger.error("Mask must be completely filled...")
            return None

        mask = mask / mask.sum(dim=0, keepdim=True)
        conds: list[torch.Tensor] = []

        for i in range(1, num_conds):
            c = kwargs[f"cond_{i}"][0][0].to(device=device, dtype=dtype)
            if (dim := math.ceil(c.shape[0] / 512) * 512) > 512:
                c = torch.nn.functional.pad(c, (0, 0, 0, dim - c.shape[0]))
            conds.append(c)

        num_tokens = [cond.shape[1] for cond in conds]

        SelfCrossAttention.couple_orig_forward = SelfCrossAttention.forward

        @wraps(SelfCrossAttention.couple_orig_forward)
        @torch.inference_mode()
        def couple_forward(
            self: "SelfCrossAttention",
            x: torch.Tensor,
            context: torch.Tensor,
            rope_emb: torch.Tensor,
            transformer_options: dict = {},
        ):
            cond_or_unconds = transformer_options.get("cond_or_uncond", None)

            if self.is_SelfAttn:
                if separation_mode != "Latent" or not cond_or_unconds:
                    return self.couple_orig_forward(
                        x=x,
                        context=context,
                        rope_emb=rope_emb,
                        transformer_options=transformer_options,
                    )

                # ===== Latent mode: self-attention regional isolation =====
                # Run self-attention independently per region and mask-blend
                # to prevent features from bleeding across regions
                num_chunks = len(cond_or_unconds)
                batch_size = x.shape[0] // num_chunks
                x_chunks = x.chunk(num_chunks, dim=0)

                num_cond_regions = len(conds)
                cond_mask = mask[1:]  # skip base mask
                cond_mask = cond_mask / cond_mask.sum(dim=0, keepdim=True).clamp(min=1e-6)

                outputs = []
                for idx, cond_or_uncond in enumerate(cond_or_unconds):
                    if cond_or_uncond == 1:
                        # uncond: normal self-attention
                        out_i = self.couple_orig_forward(
                            x=x_chunks[idx],
                            context=context,
                            rope_emb=rope_emb,
                            transformer_options=transformer_options,
                        )
                        outputs.append(out_i)
                    else:
                        # cond: run self-attn per region, mask-blend
                        region_outputs = []
                        for _ in range(num_cond_regions):
                            out_r = self.couple_orig_forward(
                                x=x_chunks[idx],
                                context=context,
                                rope_emb=rope_emb,
                                transformer_options=transformer_options,
                            )
                            region_outputs.append(out_r)

                        seq_len = region_outputs[0].shape[1]
                        stacked = torch.stack(region_outputs, dim=0)

                        mask_downsample = get_dit_mask(
                            cond_mask, seq_len, width, height,
                            patch_size=dit.patch_spatial,
                        )

                        masked_output = (stacked * mask_downsample).sum(dim=0)
                        outputs.append(masked_output)

                return torch.cat(outputs, dim=0)

            if context is None or not cond_or_unconds:
                return self.couple_orig_forward(
                    x=x,
                    context=context,
                    rope_emb=rope_emb,
                    transformer_options=transformer_options,
                )

            num_chunks = len(cond_or_unconds)
            batch_size = x.shape[0] // num_chunks

            x_chunks = x.chunk(num_chunks, dim=0)

            context_3d = context.squeeze(1)
            if (dim := math.ceil(context_3d.shape[1] / 512) * 512) > 512:
                context_3d = torch.nn.functional.pad(
                    context_3d,
                    (0, 0, 0, dim - context_3d.shape[1]),
                )

            ctx_seq_len = context_3d.shape[-2]
            context_chunks = context_3d.chunk(num_chunks, dim=0)

            lcm_tokens = lcm_for_list(num_tokens + [ctx_seq_len])
            assert lcm_tokens in (512, 1024), "Your prompt is way too long..."

            # Pre-compute per-cond context tensors for both modes
            cond_contexts = [
                cond.repeat(batch_size, lcm_tokens // cond.shape[-2], 1)
                for cond in conds
            ]

            if separation_mode == "Latent":
                # ===== Latent mode: independent forward per region =====
                # Each region's x only sees its own cond, achieving complete isolation
                # Note: conds has (num_conds - 1) elements; mask has num_conds (incl. base)
                # We use only the cond masks (skip base at index 0)
                num_cond_regions = len(conds)  # = num_conds - 1
                cond_mask = mask[1:]  # skip base mask, shape [num_cond_regions, h, w]
                # Re-normalize cond masks so they sum to 1 along dim 0
                cond_mask = cond_mask / cond_mask.sum(dim=0, keepdim=True).clamp(min=1e-6)

                outputs = []

                for idx, cond_or_uncond in enumerate(cond_or_unconds):
                    if cond_or_uncond == 1:
                        # uncond: run original forward with original context
                        c_target = context_chunks[idx].repeat(
                            1, lcm_tokens // ctx_seq_len, 1
                        ).to(dtype=x_chunks[idx].dtype)
                        out_uncond = self.couple_orig_forward(
                            x_chunks[idx],
                            context=c_target,
                            rope_emb=rope_emb,
                            transformer_options=transformer_options,
                        )
                        outputs.append(out_uncond)
                    else:
                        # cond: run independent forward for each region
                        region_outputs = []
                        for cond_idx in range(num_cond_regions):
                            single_ctx = cond_contexts[cond_idx].to(dtype=x_chunks[idx].dtype)
                            out_i = self.couple_orig_forward(
                                x_chunks[idx],
                                context=single_ctx,
                                rope_emb=rope_emb,
                                transformer_options=transformer_options,
                            )
                            region_outputs.append(out_i)

                        # Stack and mask-blend
                        seq_len = region_outputs[0].shape[1]
                        stacked = torch.stack(region_outputs, dim=0)  # [num_cond_regions, batch_size, seq_len, dim]

                        mask_downsample = get_dit_mask(
                            cond_mask, seq_len, width, height, patch_size=dit.patch_spatial
                        )

                        masked_output = (stacked * mask_downsample).sum(dim=0)
                        outputs.append(masked_output)

                return torch.cat(outputs, dim=0)

            # ===== Attention mode: original shared forward with mask blending =====
            conds_tensor = torch.cat(cond_contexts, dim=0)

            new_x = []
            new_context = []

            for idx, cond_or_uncond in enumerate(cond_or_unconds):
                c_target = context_chunks[idx].repeat(1, lcm_tokens // ctx_seq_len, 1)
                if cond_or_uncond == 1:
                    new_x.append(x_chunks[idx])
                    new_context.append(c_target)
                else:
                    new_x.append(x_chunks[idx].repeat(num_conds, 1, 1))
                    new_context.append(torch.cat([c_target, conds_tensor], dim=0))

            x_in = torch.cat(new_x, dim=0)
            ctx_in = torch.cat(new_context, dim=0).to(dtype=x_in.dtype)

            out = self.couple_orig_forward(
                x_in,
                context=ctx_in,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

            seq_len: int = out.shape[1]

            mask_downsample = get_dit_mask(
                mask, seq_len, width, height, patch_size=dit.patch_spatial
            )

            mask_downsample = sharpen_mask(
                mask_downsample, mask_mode, mask_temperature
            )

            outputs = []
            pos = 0

            for idx, cond_or_uncond in enumerate(cond_or_unconds):
                if cond_or_uncond == 1:
                    outputs.append(out[pos : pos + batch_size])
                    pos += batch_size
                else:
                    chunk = out[pos : pos + num_conds * batch_size]
                    chunk = chunk.view(num_conds, batch_size, seq_len, -1)

                    masked_output = (chunk * mask_downsample).sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * batch_size

            return torch.cat(outputs, dim=0)

        couple_forward._couple = True
        SelfCrossAttention.forward = couple_forward

        return model

    @staticmethod
    def unpatch():
        if hasattr(SelfCrossAttention, "couple_orig_forward"):
            if getattr(SelfCrossAttention.forward, "_couple", False):
                SelfCrossAttention.forward = SelfCrossAttention.couple_orig_forward
            del SelfCrossAttention.couple_orig_forward
