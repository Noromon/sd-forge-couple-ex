import gradio as gr

from modules.shared import opts

from .gr_version import js
from .ui_adv import advanced_ui
from .ui_funcs import on_pull
from .ui_masks import CoupleMaskData
from .ui_tile import tile_ui


class CoupleDataTransfer:
    """Handle sending data from t2i/i2i to i2i/t2i"""

    T2I_MASK: CoupleMaskData = None
    I2I_MASK: CoupleMaskData = None

    T2I_ADV_DATA: gr.JSON = None
    T2I_ADV_PASTE: gr.Textbox = None
    T2I_ADV_PULL: gr.Button = None

    I2I_ADV_DATA: gr.JSON = None
    I2I_ADV_PASTE: gr.Textbox = None
    I2I_ADV_PULL: gr.Button = None

    A_HOOKED: bool = False
    M_HOOKED: bool = False

    @classmethod
    def hook_adv(cls):
        assert not any(
            [
                cls.T2I_ADV_DATA is None,
                cls.T2I_ADV_PASTE is None,
                cls.T2I_ADV_PULL is None,
                cls.I2I_ADV_DATA is None,
                cls.I2I_ADV_PASTE is None,
                cls.I2I_ADV_PULL is None,
            ]
        )

        cls.I2I_ADV_PULL.click(
            fn=on_pull, inputs=cls.T2I_ADV_DATA, outputs=cls.I2I_ADV_PASTE
        )

        cls.T2I_ADV_PULL.click(
            fn=on_pull, inputs=cls.I2I_ADV_DATA, outputs=cls.T2I_ADV_PASTE
        )

        cls.A_HOOKED = True

    @classmethod
    def hook_mask(cls):
        assert cls.T2I_MASK is not None
        assert cls.I2I_MASK is not None

        cls.T2I_MASK.opposite = cls.I2I_MASK
        cls.I2I_MASK.opposite = cls.T2I_MASK

        cls.M_HOOKED = True

    @classmethod
    def webui_setup_done(cls) -> bool:
        return cls.A_HOOKED and cls.M_HOOKED


def couple_ui(script, is_img2img: bool, title: str):
    m: str = "i2i" if is_img2img else "t2i"

    with gr.Accordion(
        label=title,
        elem_id=f"forge_couple_{m}",
        open=False,
    ):
        if is_img2img and not getattr(opts, "fc_no_tile", False):
            tab1 = gr.Tab(label="Regions")
            tab1.__enter__()

        with gr.Row():
            with gr.Column(
                elem_classes="fc-checkbox",
                scale=2,
            ):
                enable = gr.Checkbox(False, label="Enable")
                disable_hr = gr.Checkbox(True, label="Compatibility")

            mode = gr.Radio(
                ["Basic", "Advanced", "Mask"],
                label="Region Assignment",
                value="Basic",
                scale=3,
            )

            separator = gr.Textbox(
                value="",
                label="Couple Separator",
                lines=1,
                max_lines=1,
                placeholder="\\n",
                elem_classes="fc_separator",
                scale=1,
            )

        with gr.Group(visible=True, elem_classes="fc_bsc") as basic_settings:
            with gr.Row():
                direction = gr.Radio(
                    ["Horizontal", "Vertical"],
                    label="Tile Direction",
                    value="Horizontal",
                    scale=2,
                )

                placeholder = gr.Group(visible=False, elem_classes="fc_placeholder")
                placeholder.do_not_save_to_config = True

                background = gr.Radio(
                    ["None", "First Line", "Last Line"],
                    label="Global Effect",
                    value="None",
                    scale=3,
                    elem_classes="fc_global_effect",
                )

                background_weight = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                    label="Global Effect Weight",
                    scale=1,
                    interactive=False,
                )

            def on_background_change(choice: str):
                return gr.update(interactive=(choice != "None"))

            background.change(
                on_background_change,
                background,
                background_weight,
                show_progress="hidden",
            ).success(None, **js(f'() => {{ ForgeCouple.onBackgroundChange("{m}"); }}'))

        with gr.Group(visible=False, elem_classes="fc_adv") as adv_settings:
            preview_btn, preview_res, mapping_paste_field, mapping, pull_btn = (
                advanced_ui(is_img2img, m, mode)
            )

            if not CoupleDataTransfer.webui_setup_done():
                if is_img2img:
                    CoupleDataTransfer.I2I_ADV_DATA = mapping
                    CoupleDataTransfer.I2I_ADV_PASTE = mapping_paste_field
                    CoupleDataTransfer.I2I_ADV_PULL = pull_btn
                    CoupleDataTransfer.hook_adv()  # img2img always happens after txt2img

                else:
                    CoupleDataTransfer.T2I_ADV_DATA = mapping
                    CoupleDataTransfer.T2I_ADV_PASTE = mapping_paste_field
                    CoupleDataTransfer.T2I_ADV_PULL = pull_btn

        with gr.Group(visible=False, elem_classes="fc_msk") as msk_settings:
            couple_mask = CoupleMaskData(is_img2img)
            couple_mask.mask_ui(preview_btn, preview_res, mode)

            if not CoupleDataTransfer.webui_setup_done():
                if is_img2img:
                    CoupleDataTransfer.I2I_MASK = couple_mask
                    CoupleDataTransfer.hook_mask()  # img2img always happens after txt2img
                else:
                    CoupleDataTransfer.T2I_MASK = couple_mask

        with gr.Accordion(
            label="Common Prompts",
            elem_id=f"forge_couple_cmp_{m}",
            open=False,
        ):
            with gr.Row():
                common_parser = gr.Radio(
                    ("off", "{ }", "< >"), label="Syntax", value="{ }", scale=3
                )
                def_in_prompt = gr.Checkbox(
                    True, label="Include Definitions in Prompt", scale=3
                )
                common_debug = gr.Checkbox(False, label="Debug", scale=1)
                common_debug.do_not_save_to_config = True

        with gr.Accordion(
            label="Separation Mode",
            elem_id=f"forge_couple_sep_{m}",
            open=False,
        ):
            with gr.Row():
                separation_mode = gr.Radio(
                    ["Attention", "Latent", "Hybrid", "Independent"],
                    label="Separation Method",
                    value="Attention",
                    scale=2,
                    info=(
                        "Attention: mask blending after shared forward | "
                        "Latent: independent forward per region (stronger isolation) | "
                        "Hybrid: global self-attn + per-region cross-attn/MLP (1x time, no seam) | "
                        "Independent: n full passes with per-region LoRA weights, fully isolated (n x time)"
                    ),
                )

            with gr.Group(visible=True, elem_classes="fc_sharp") as sharp_settings:
                with gr.Row():
                    mask_mode = gr.Radio(
                        ["Soft", "Hard", "Temperature"],
                        label="Mask Sharpening",
                        value="Soft",
                        scale=2,
                        info="Soft: original blending | Hard: winner-take-all | Temperature: adjustable",
                    )
                    mask_temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        label="Temperature",
                        info="Lower = sharper separation (only used in Temperature mode)",
                        scale=3,
                        interactive=False,
                    )

            with gr.Group(visible=False, elem_classes="fc_blend") as blend_settings:
                with gr.Row():
                    blend_mode = gr.Radio(
                        ["Hard", "Feather"],
                        label="Region Blend",
                        value="Hard",
                        scale=2,
                        info=(
                            "How per-region results are merged in latent space. "
                            "Hard: crisp region boundaries | Feather: soft transition band"
                        ),
                    )
                    feather_width = gr.Slider(
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=8,
                        label="Feather Width (px)",
                        info="Transition width in pixels (only used in Feather mode)",
                        scale=3,
                        interactive=False,
                    )

            with gr.Group(visible=False, elem_classes="fc_hybrid") as hybrid_settings:
                with gr.Row():
                    boundary_mode = gr.Radio(
                        ["Hard", "Soft"],
                        label="Boundary Transition",
                        value="Soft",
                        scale=2,
                        info=(
                            "How regions are routed at the token level. "
                            "Hard: each pixel belongs to exactly one region | "
                            "Soft: Gaussian-blended transition band (smooths seams)"
                        ),
                    )
                    soft_width = gr.Slider(
                        minimum=0,
                        maximum=128,
                        step=1,
                        value=16,
                        label="Transition Width (px)",
                        info="Width of the blended boundary band (only used in Soft mode)",
                        scale=3,
                        interactive=True,  # default boundary_mode is "Soft" -> active on load
                    )
                    soft_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Soft Strength",
                        info="1.0 = fully blurred weights | 0.0 = hard argmax (only used in Soft mode)",
                        scale=3,
                        interactive=True,  # default boundary_mode is "Soft" -> active on load
                    )

            def on_mask_mode_change(choice: str):
                return gr.update(interactive=(choice == "Temperature"))

            mask_mode.change(
                on_mask_mode_change,
                mask_mode,
                mask_temperature,
                show_progress="hidden",
            )

            def on_blend_mode_change(choice: str):
                return gr.update(interactive=(choice == "Feather"))

            blend_mode.change(
                on_blend_mode_change,
                blend_mode,
                feather_width,
                show_progress="hidden",
            )

            def on_boundary_mode_change(choice: str):
                return [
                    gr.update(interactive=(choice == "Soft")),
                    gr.update(interactive=(choice == "Soft")),
                ]

            boundary_mode.change(
                on_boundary_mode_change,
                boundary_mode,
                [soft_width, soft_strength],
                show_progress="hidden",
            )

            def on_separation_mode_change(choice: str):
                return [
                    gr.update(visible=(choice == "Attention")),
                    gr.update(visible=(choice in ("Independent",))),
                    gr.update(visible=(choice == "Hybrid")),
                ]

            separation_mode.change(
                on_separation_mode_change,
                separation_mode,
                [sharp_settings, blend_settings, hybrid_settings],
                show_progress="hidden",
            )

        def on_mode_change(choice: str):
            return [
                gr.update(visible=(choice in ("Basic", "Mask"))),
                gr.update(visible=(choice == "Basic")),
                gr.update(visible=(choice == "Advanced")),
                gr.update(visible=(choice == "Mask")),
                gr.update(visible=(choice == "Mask")),
            ]

        mode.change(
            on_mode_change,
            mode,
            [basic_settings, direction, adv_settings, msk_settings, placeholder],
            show_progress="hidden",
        ).success(fn=None, **js(f'() => {{ ForgeCouple.preview("{m}"); }}'))

        script.paste_field_names = []
        script.infotext_fields = [
            (enable, "forge_couple"),
            (disable_hr, "forge_couple_compatibility"),
            (mode, "forge_couple_mode"),
            (separator, "forge_couple_separator"),
            (direction, "forge_couple_direction"),
            (background, "forge_couple_background"),
            (background_weight, "forge_couple_background_weight"),
            (mapping_paste_field, "forge_couple_mapping"),
            (common_parser, "forge_couple_common_parser"),
            (separation_mode, "forge_couple_separation_mode"),
            (mask_mode, "forge_couple_mask_mode"),
            (mask_temperature, "forge_couple_mask_temperature"),
            (blend_mode, "forge_couple_blend_mode"),
            (feather_width, "forge_couple_feather_width"),
            (boundary_mode, "forge_couple_boundary_mode"),
            (soft_width, "forge_couple_soft_width"),
            (soft_strength, "forge_couple_soft_strength"),
        ]

        for comp, name in script.infotext_fields:
            comp.do_not_save_to_config = True
            script.paste_field_names.append(name)

        if is_img2img and not getattr(opts, "fc_no_tile", False):
            tab1.__exit__()
            with gr.Tab(label="Tiles"):
                tile_args = tile_ui()

        else:
            tile_args = [gr.State(None)] * 6

    return [
        enable,
        disable_hr,
        mode,
        separator,
        direction,
        background,
        background_weight,
        mapping,
        common_parser,
        common_debug,
        def_in_prompt,
        separation_mode,
        mask_mode,
        mask_temperature,
        blend_mode,
        feather_width,
        boundary_mode,
        soft_width,
        soft_strength,
        *tile_args,
    ], couple_mask.get_masks
