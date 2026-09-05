import re
from json import dumps
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing as P

import torch

from lib_couple import settings  # noqa
from lib_couple.attention_couple import AttentionCouple
from lib_couple.gr_version import js
from lib_couple.logging import logger
from lib_couple.mapping import (
    advanced_mapping,
    basic_mapping,
    empty_tensor,
    mask_mapping,
)
from lib_couple.tile_funcs import calculate_tiles
from lib_couple.ui import couple_ui
from lib_couple.ui_funcs import validate_mapping

try:
    from lib_couple.anima import AttentionCoupleAnima
except ImportError:
    is_neo = False
else:
    is_neo = True

from modules import scripts, shared

VERSION = "7.3.0"

UI_CACHES: dict[bool, tuple[list, Callable]] = {}


class ForgeCouple(scripts.Script):

    # <lora:name:...> style extra-network tags (same grammar as the framework)
    re_extra_net = re.compile(r"<(\w+):([^>]+)>")

    # Innermost-first variant: content may not contain brackets, so a tag nested inside
    # another one is matched before its outer wrapper.
    re_lora_inner = re.compile(r"<(\w+):([^<>]+)>")

    # Placeholder markers used to shield <lora:...> tags from common-prompt parsing.
    re_ph = re.compile("\ue000(\\d+)\ue001")

    def __init__(self):
        self.is_img2img: bool
        self.couples: list
        self.get_mask: Callable
        self.is_hr: bool

        self.valid: bool
        """
        Since raising error within Extensions does NOT cancel the generation,
        the only way is to forcefully interrupt during generation...
        """

        self.tile_idx: int
        self.tiles: list[str] = []

        # Independent (true latent) mode state
        self._fc_raw_prompt: str | None = None
        self._fc_negative_text: str = ""
        self._fc_separator: str = "\n"

    def title(self):
        return "Forge Couple"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.is_img2img = is_img2img
        if is_img2img in UI_CACHES:
            comps, func = UI_CACHES[is_img2img]
        else:
            comps, func = couple_ui(self, is_img2img, f"{self.title()} v{VERSION}")
            UI_CACHES[is_img2img] = (comps, func)
        self.get_mask = func
        return comps

    def after_component(self, component, **kwargs):
        if (elem_id := kwargs.get("elem_id", None)) is not None:
            if elem_id in ("txt2img_width", "txt2img_height"):
                component.change(None, **js('() => { ForgeCouple.preview("t2i"); }'))
            elif elem_id in ("img2img_width", "img2img_height"):
                component.change(None, **js('() => { ForgeCouple.preview("i2i"); }'))

    def setup(self, p, *args, **kwargs):
        self.is_hr = False
        if not self.is_img2img or getattr(shared.opts, "fc_no_tile", False):
            return

        if calculate_tiles(self, (p, *args)) is None:
            self.invalidate(p)

        self.tile_idx = -1

    def before_process(self, p, *args, **kwargs):
        # Restore any Hybrid state left over from a previous generation (undo in-place
        # Anima block forwards / UNet clone patches) so it never leaks into this run.
        try:
            if getattr(p.sd_model, "_hybrid_saved_unet", None) is not None:
                from lib_couple.hybrid import restore_hybrid

                restore_hybrid(p.sd_model)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Hybrid] Failed to restore previous state: {e}")

        if not self.tiles:
            return

        self.tile_idx += 1
        p.prompt = self.tiles[self.tile_idx]
        debug: bool = args[-1]

        if debug:
            print("")
            logger.info(f"[Tile Debug]\n{p.prompt}\n")

    def before_hr(self, p: "P", *args, **kwargs):
        self.is_hr = True
        if is_neo and p.sd_model.model_config.huggingface_repo.endswith("Anima"):
            AttentionCoupleAnima.unpatch()

    def _is_tile(self) -> bool:
        return self.is_img2img and len(self.tiles) > 0

    @staticmethod
    def parse_common_prompt(
        prompt: str,
        brackets: tuple[str],
        def_in_prompt: bool,
    ) -> str:
        common_prompts: dict[str, str] = {}
        op, cs = brackets

        pattern = rf"{op}([^{op}{cs}]+?):([^{op}{cs}]+?){cs}"
        matches = list(re.finditer(pattern, prompt))
        for m in matches:
            key: str = m.group(1).strip()
            val: str = m.group(2).strip()
            prompt = prompt.replace(m.group(0), val if def_in_prompt else "")
            common_prompts.update({key: val})

        pattern = rf"{op}([^{op}{cs}]+?){cs}"
        matches = list(re.finditer(pattern, prompt))
        for m in matches:
            key: str = m.group(1).strip()
            if key in common_prompts:
                prompt = prompt.replace(m.group(0), common_prompts[key])

        return prompt

    def invalidate(self, p):
        self.valid = False
        p.extra_generation_params.update({"forge_couple": "ERROR"})
        if shared.opts.fc_do_interrupt:
            shared.state.interrupt()

    # ===== Independent (true latent) mode helpers =====

    @classmethod
    def _parse_lora_tag(cls, args_str: str) -> tuple[str, float, float]:
        """Parse the argument string of a <lora:...> tag into (name, unet_w, te_w)."""
        items = args_str.split(":")
        positional: list[str] = []
        named: dict[str, str] = {}

        for item in items:
            parts = item.split("=", 2) if isinstance(item, str) else [item]
            if len(parts) == 2:
                named[parts[0].strip()] = parts[1].strip()
            else:
                positional.append(item)

        name = positional[0].strip() if positional else ""

        def _pos(idx: int, default: float | None) -> float | None:
            if idx < len(positional) and positional[idx].strip():
                try:
                    return float(positional[idx])
                except ValueError:
                    return default
            return default

        te = _pos(1, None)
        if te is None:
            te = float(named["te"]) if "te" in named else 1.0

        unet = _pos(2, None)
        if unet is None:
            unet = float(named["unet"]) if "unet" in named else te

        return name, unet, te

    def _extract_loras(self, text: str) -> tuple[list[tuple[str, float, float]], str]:
        """Pull every <lora:...> tag out of `text`; return (loras, cleaned_text)."""
        loras: list[tuple[str, float, float]] = []

        def _repl(m):
            if m.group(1).lower() == "lora":
                try:
                    loras.append(self._parse_lora_tag(m.group(2)))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[Independent] Failed to parse LoRA tag {m.group(0)}: {e}")
            return ""

        cleaned = self.re_extra_net.sub(_repl, text)
        return loras, cleaned.strip()

    @staticmethod
    def _merge_loras(*loras_lists) -> list[tuple[str, float, float]]:
        """Merge LoRA lists (later entries override earlier ones by name)."""
        merged: dict[str, tuple[str, float, float]] = {}
        for lst in loras_lists:
            for name, unet_w, te_w in lst or []:
                if name:
                    merged[name] = (name, unet_w, te_w)
        return list(merged.values())

    def _prepare_independent_regions(
        self,
        raw_prompt: str,
        separator: str,
        common_parser: str,
        def_in_prompt: bool,
    ) -> tuple[list[str], list[list[tuple[str, float, float]]]]:
        """Split the raw (LoRA-tagged) prompt into per-region texts + LoRA sets.

        <lora:...> tags are first shielded with bracket-free placeholder tokens so they
        can never be mangled by common-prompt parsing (critical for nested `< >`
        definitions). Placeholders then flow through definition/reference expansion
        naturally, which makes a definition's LoRAs belong to every region that
        references it (and to the defining region when its content is inlined).
        """
        chunks = [c.strip() for c in raw_prompt.split(separator)]

        # Step 1: shield <lora:...> tags with unique placeholders. Innermost-first so a
        # LoRA nested inside another tag (e.g. within a `< >` definition) survives.
        loras_by_ph: dict[str, tuple[str, float, float]] = {}
        counter = [0]

        def _ph_repl(m):
            if m.group(1).lower() != "lora":
                return m.group(0)
            try:
                entry = self._parse_lora_tag(m.group(2))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[Independent] Failed to parse LoRA tag {m.group(0)}: {e}")
                return ""
            ph = f"\ue000{counter[0]}\ue001"
            counter[0] += 1
            loras_by_ph[ph] = entry
            return ph

        def _shield(text: str) -> str:
            prev = None
            while text != prev:
                prev = text
                text = self.re_lora_inner.sub(_ph_repl, text)
            return text

        chunks = [_shield(c) for c in chunks]

        def _resolve(text: str) -> tuple[str, list[tuple[str, float, float]]]:
            """Strip placeholders from `text`; return (clean_text, loras first-seen order)."""
            seen: set[str] = set()
            loras: list[tuple[str, float, float]] = []

            def _sub(m):
                ph = f"\ue000{m.group(1)}\ue001"
                if ph not in seen and ph in loras_by_ph:
                    seen.add(ph)
                    loras.append(loras_by_ph[ph])
                return ""

            clean = self.re_ph.sub(_sub, text).strip()
            return clean, loras

        if common_parser not in ("{ }", "< >"):
            # No common prompts: just resolve each chunk's own LoRAs.
            final_chunks: list[str] = []
            loras_per_region: list[list[tuple[str, float, float]]] = []
            for text in chunks:
                clean, loras = _resolve(text)
                final_chunks.append(clean)
                loras_per_region.append(self._merge_loras(loras))
            return final_chunks, loras_per_region

        op, cs = common_parser.split(" ")
        def_pat = re.compile(rf"{op}([^{op}{cs}]+?):([^{op}{cs}]+?){cs}")
        ref_pat = re.compile(rf"{op}([^{op}{cs}]+?){cs}")

        # Step 2: definitions (values are already placeholder-ized, so nested tags can't
        # break the bracket matching).
        clean_defs: dict[str, str] = {}
        working: list[str] = []
        for chunk in chunks:
            def _def_repl(m):
                key = m.group(1).strip()
                value = m.group(2).strip()
                clean_defs[key] = value
                return value if def_in_prompt else ""

            working.append(def_pat.sub(_def_repl, chunk))

        # Step 3: references to known keys.
        final_chunks: list[str] = []
        for text in working:
            def _ref_repl(m):
                key = m.group(1).strip()
                if key in clean_defs:
                    return clean_defs[key]
                return m.group(0)  # unknown key -- leave untouched

            final_chunks.append(ref_pat.sub(_ref_repl, text))

        # Step 4: resolve placeholders -> per-region loras + clean texts.
        loras_per_region = []
        for i, text in enumerate(final_chunks):
            clean, loras = _resolve(text)
            final_chunks[i] = clean
            loras_per_region.append(self._merge_loras(loras))

        return final_chunks, loras_per_region

    def _do_independent(
        self,
        p: "P",
        fc_args: dict,
        width: int,
        height: int,
        line_count: int,
        separator: str,
        common_parser: str,
        def_in_prompt: bool,
        blend_mode: str,
        feather_width: float,
        x_init: torch.Tensor,
        noise_i2i: torch.Tensor | None,
    ):
        """Run one full independent inference pass per region and install a stub sampler.

        Args:
            x_init: the initial latent (noise for t2i, init_latent for i2i/HR)
            noise_i2i: fresh noise for img2img-style passes (None for pure t2i)
        """
        from lib_couple.independent import PrecomputedSampler, run_independent_passes

        # Region masks [n, H, W] (weights; normalized during blending).
        masks = torch.stack(
            [fc_args[f"mask_{i}"] for i in range(1, line_count + 1)], dim=0
        )  # [n, 1, H, W] or [n, H, W] depending on mapping output
        if masks.dim() == 4:
            masks = masks.squeeze(1)  # -> [n, H, W]

        if masks.sum(dim=0).min().item() <= 0.0:
            logger.error("[Independent] Mask must be completely filled...")
            self.invalidate(p)
            return

        # Per-region texts + LoRAs (from the raw prompt captured in before_process_batch).
        region_texts, loras_per_region = self._prepare_independent_regions(
            self._fc_raw_prompt or "", separator, common_parser, def_in_prompt
        )

        if len(region_texts) != line_count:
            logger.error(
                f"[Independent] Region count mismatch "
                f"({len(region_texts)} / {line_count})..."
            )
            self.invalidate(p)
            return

        # Global (negative-prompt) LoRAs apply to every region.
        neg_loras, _ = self._extract_loras(self._fc_negative_text or "")
        final_loras_per_region = [
            self._merge_loras(neg_loras, loras_per_region[i]) for i in range(line_count)
        ]

        # Determine if this is an img2img-style pass (i2i first pass OR HR pass).
        is_i2i_pass = self.is_hr or (self.is_img2img and x_init is not None)

        result = run_independent_passes(
            p=p,
            is_i2i=is_i2i_pass,
            x_init=x_init,
            noise_i2i=noise_i2i,
            region_texts=region_texts,
            negative_text=(self._fc_negative_text or ""),
            region_masks=masks,
            loras_per_region=final_loras_per_region,
            width=width,
            height=height,
            blend_mode=blend_mode,
            feather_width=feather_width,
        )

        if result is None:
            self.invalidate(p)
            return

        # Replace the sampler so the framework's sampling call becomes a passthrough.
        p.sampler = PrecomputedSampler(result)
        logger.info("[Independent] Installed pre-computed sampler")

    def after_extra_networks_activate(
        self,
        p: "P",
        enable: bool,
        disable_hr: bool,
        mode: str,
        separator: str,
        direction: str,
        background: str,
        background_weight: float,
        mapping: list,
        common_parser: str,
        common_debug: bool,
        def_in_prompt: bool,
        *args,
        **kwargs,
    ):
        self.couples = None
        if not enable:
            return

        if self._is_tile():
            return

        separator = separator.replace("\\n", "\n").replace("\\t", " ")
        if not separator.strip():
            separator = "\n"
        self._fc_separator = separator

        prompts: str = kwargs["prompts"][0]

        if common_parser in ("{ }", "< >"):
            prompts = self.parse_common_prompt(
                prompts,
                common_parser.split(" "),
                def_in_prompt,
            )
            if common_debug:
                print("")
                logger.info(f"[Common Prompts Debug]\n{prompts}\n")

        couples: list[str] = [chunk.strip() for chunk in prompts.split(separator)]

        match mode:
            case "Basic":
                if len(couples) < (3 - int(background == "None")):
                    ratio = f"{len(couples)} / {3 - int(background == 'None')}"
                    logger.error(f"Not Enough Lines in Prompt... [{ratio}]")
                    self.invalidate(p)
                    return

            case "Mask":
                mapping: list = self.get_mask() or mapping
                assert isinstance(mapping[0], dict)
                if not mapping:
                    logger.error("No Mapping...?")
                    self.invalidate(p)
                    return

                required: int = len(mapping) + int(background != "None")
                if len(couples) != required:
                    ratio = f"{len(couples)} / {required}"
                    logger.error(f"Number of Couples and Masks mismatched... [{ratio}]")
                    self.invalidate(p)
                    return

            case "Advanced":
                assert isinstance(mapping[0], list)
                if not mapping:
                    logger.error("No Mapping...?")
                    self.invalidate(p)
                    return

                if not validate_mapping(mapping, True):
                    self.invalidate(p)
                    return

                if len(couples) != len(mapping):
                    ratio = f"{len(couples)} / {len(mapping)}"
                    logger.error(f"Number of Couples and Masks mismatched... [{ratio}]")
                    self.invalidate(p)
                    return

        # ===== Infotext =====
        fc_param: dict = {}

        fc_param["forge_couple"] = True
        fc_param["forge_couple_compatibility"] = disable_hr
        fc_param["forge_couple_mode"] = mode
        fc_param["forge_couple_separator"] = separator.replace("\n", "\\n")
        if mode == "Basic":
            fc_param["forge_couple_direction"] = direction
        if mode == "Advanced":
            fc_param["forge_couple_mapping"] = dumps(mapping)
        else:
            fc_param["forge_couple_background"] = background
            fc_param["forge_couple_background_weight"] = background_weight
        fc_param["forge_couple_common_parser"] = common_parser
        fc_param["forge_couple_def_in_prompt"] = def_in_prompt

        # ===== Separation Mode & Mask Sharpening (infotext) =====
        # args layout (after the 9 named params):
        #   common_parser(0), common_debug(1), def_in_prompt(2),
        #   separation_mode(3), mask_mode(4), mask_temperature(5),
        #   blend_mode(6), feather_width(7), boundary_mode(8), soft_width(9),
        #   soft_strength(10), use_tile(11), ...
        if len(args) >= 4:
            fc_param["forge_couple_separation_mode"] = args[3]
        if len(args) >= 5:
            fc_param["forge_couple_mask_mode"] = args[4]
        if len(args) >= 6:
            fc_param["forge_couple_mask_temperature"] = args[5]
        if len(args) >= 7:
            fc_param["forge_couple_blend_mode"] = args[6]
        if len(args) >= 8:
            fc_param["forge_couple_feather_width"] = args[7]
        if len(args) >= 9:
            fc_param["forge_couple_boundary_mode"] = args[8]
        if len(args) >= 10:
            fc_param["forge_couple_soft_width"] = args[9]
        if len(args) >= 11:
            fc_param["forge_couple_soft_strength"] = args[10]

        p.extra_generation_params.update(fc_param)
        # ===== Infotext =====

        self.couples = couples
        self.valid = True

    def before_process_batch(self, p: "P", *args, **kwargs):
        if is_neo and p.sd_model.model_config.huggingface_repo.endswith("Anima"):
            AttentionCoupleAnima.unpatch()

        # Capture the raw prompt (still containing <lora:...> tags) for Independent mode.
        try:
            self._fc_raw_prompt = kwargs["prompts"][0]
        except Exception:  # noqa: BLE001
            self._fc_raw_prompt = None
        try:
            negs = p.negative_prompts or []
            self._fc_negative_text = negs[0] if negs else (p.negative_prompt or "")
        except Exception:  # noqa: BLE001
            self._fc_negative_text = ""

    def process_before_every_sampling(
        self,
        p: "P",
        enable: bool,
        disable_hr: bool,
        mode: str,
        separator: str,
        direction: str,
        background: str,
        background_weight: float,
        mapping: list,
        *args,
        **kwargs,
    ):
        if (not enable) or (self.couples is None) or (not self.valid):
            return

        if self._is_tile():
            return

        if disable_hr and self.is_hr:
            return

        if getattr(p, "_ad_inner", False):
            return

        # ===== Separation Mode & Mask Sharpening =====
        # args layout (after the 9 named params):
        #   common_parser(0), common_debug(1), def_in_prompt(2),
        #   separation_mode(3), mask_mode(4), mask_temperature(5),
        #   blend_mode(6), feather_width(7), boundary_mode(8), soft_width(9),
        #   soft_strength(10), use_tile(11), tile_h(12), ...
        separation_mode: str = "Attention"
        mask_mode: str = "Soft"
        mask_temperature: float = 1.0
        blend_mode: str = "Hard"
        feather_width: float = 0.0
        boundary_mode: str = "Soft"
        soft_width: float = 16.0
        soft_strength: float = 1.0
        if len(args) >= 4:
            separation_mode = args[3] if args[3] else "Attention"
        if len(args) >= 5:
            mask_mode = args[4] if args[4] else "Soft"
        if len(args) >= 6:
            mask_temperature = args[5] if args[5] is not None else 1.0
        if len(args) >= 7:
            blend_mode = args[6] if args[6] else "Hard"
        if len(args) >= 8:
            feather_width = float(args[7]) if args[7] is not None else 0.0
        if len(args) >= 9:
            boundary_mode = args[8] if args[8] else "Soft"
        if len(args) >= 10:
            soft_width = float(args[9]) if args[9] is not None else 16.0
        if len(args) >= 11:
            soft_strength = float(args[10]) if args[10] is not None else 1.0

        # ===== Init =====
        WIDTH: int = p.hr_upscale_to_x if self.is_hr else p.width
        HEIGHT: int = p.hr_upscale_to_y if self.is_hr else p.height
        IS_HORIZONTAL: bool = direction == "Horizontal"
        NO_BACKGROUND: bool = background == "None"

        LINE_COUNT: int = len(self.couples)

        if mode != "Advanced":
            BG_WEIGHT: float = 0.0 if NO_BACKGROUND else max(0.1, background_weight)

        if mode == "Basic":
            TILE_COUNT: int = LINE_COUNT - int(not NO_BACKGROUND)
            TILE_WEIGHT: float = 1.25 if NO_BACKGROUND else 1.0
            TILE_SIZE: int = (
                (WIDTH if IS_HORIZONTAL else HEIGHT) - 1
            ) // TILE_COUNT + 1
        # ===== Init =====

        # ===== Tiles =====
        match mode:
            case "Basic":
                fc_args = basic_mapping(
                    p.sd_model,
                    self.couples,
                    WIDTH,
                    HEIGHT,
                    LINE_COUNT,
                    IS_HORIZONTAL,
                    background,
                    TILE_SIZE,
                    TILE_WEIGHT,
                    BG_WEIGHT,
                )

            case "Mask":
                mapping: list[dict] = self.get_mask() or mapping

                fc_args = mask_mapping(
                    p.sd_model,
                    self.couples,
                    WIDTH,
                    HEIGHT,
                    LINE_COUNT,
                    mapping,
                    background,
                    BG_WEIGHT,
                )

            case "Advanced":
                fc_args = advanced_mapping(
                    p.sd_model, self.couples, WIDTH, HEIGHT, mapping
                )
        # ===== Tiles =====

        assert len(fc_args.keys()) // 2 == LINE_COUNT

        # ===== Independent (true latent) mode: n full passes with per-region LoRAs =====
        if separation_mode == "Independent":
            x_init = kwargs.get("x", None)
            noise_i2i = kwargs.get("noise", None)

            if x_init is None:
                logger.error("[Independent] No initial latent (kwargs['x']) available...")
                self.invalidate(p)
                return

            self._do_independent(
                p,
                fc_args,
                WIDTH,
                HEIGHT,
                LINE_COUNT,
                self._fc_separator,
                args[0] if len(args) >= 1 else "{ }",
                def_in_prompt=(args[2] if len(args) >= 3 else True),
                blend_mode=blend_mode,
                feather_width=feather_width,
                x_init=x_init,
                noise_i2i=noise_i2i,
            )
            return

        # ===== Hybrid mode: single pass, global self-attn + per-region cross-attn/MLP =====
        if separation_mode == "Hybrid":
            from lib_couple.hybrid import run_hybrid

            region_texts, loras_per_region = self._prepare_independent_regions(
                self._fc_raw_prompt or "", self._fc_separator,
                args[0] if len(args) >= 1 else "{ }",
                def_in_prompt=(args[2] if len(args) >= 3 else True),
            )

            if len(region_texts) != LINE_COUNT:
                logger.error(
                    f"[Hybrid] Region count mismatch "
                    f"({len(region_texts)} / {LINE_COUNT})..."
                )
                self.invalidate(p)
                return

            # Global (negative-prompt) LoRAs apply to every region.
            neg_loras, _ = self._extract_loras(self._fc_negative_text or "")
            final_loras_per_region = [
                self._merge_loras(neg_loras, loras_per_region[i]) for i in range(LINE_COUNT)
            ]

            if not run_hybrid(
                p=p,
                fc_args=fc_args,
                width=WIDTH,
                height=HEIGHT,
                loras_per_region=final_loras_per_region,
                region_texts=region_texts,
                boundary_mode=boundary_mode,
                soft_width=soft_width,
                soft_strength=soft_strength,
            ):
                self.invalidate(p)
            return

        unet = p.sd_model.forge_objects.unet
        base_mask = empty_tensor(HEIGHT, WIDTH)

        if is_neo and p.sd_model.model_config.huggingface_repo.endswith("Anima"):
            patched_unet = AttentionCoupleAnima.patch_dit(
                unet, base_mask, WIDTH, HEIGHT, fc_args,
                separation_mode=separation_mode,
                mask_mode=mask_mode, mask_temperature=mask_temperature,
            )
        else:
            patched_unet = AttentionCouple.patch_unet(
                unet, base_mask, fc_args,
                separation_mode=separation_mode,
                mask_mode=mask_mode, mask_temperature=mask_temperature,
            )

        if patched_unet is None:
            self.invalidate(p)
        else:
            p.sd_model.forge_objects.unet = patched_unet
