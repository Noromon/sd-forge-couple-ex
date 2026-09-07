import gradio as gr

from modules.script_callbacks import on_ui_settings
from modules.shared import OptionInfo, opts


def fc_settings():
    args = {"section": ("fc", "Forge Couple"), "category_id": "sd"}

    opts.add_option(
        "fc_hybrid_sdpa",
        OptionInfo(
            "auto",
            "Hybrid SDPA Backend (Anima DiT attention)",
            gr.Dropdown,
            {"choices": ("auto", "flash", "mem_efficient", "math", "cudnn")},
            **args,
        ).info(
            'Forces a specific scaled_dot_product_attention backend for the Anima '
            'DiT blocks during Hybrid sampling. "auto" = torch default (the chosen '
            'kernel can differ across GPU architectures, which may cause subtle '
            'cross-machine result drift). Use "math" or "mem_efficient" if results '
            'look over-saturated / smeared on one machine only.'
        ),
    )

    opts.add_option(
        "fc_hybrid_cache",
        OptionInfo(
            "last",
            "Hybrid Region Cache",
            gr.Dropdown,
            {"choices": ("off", "last")},
            **args,
        ).info(
            '"last" = keep the deep-copied region modules of the most recent '
            'generation (reused when the same model + LoRA combination is run '
            'again; any change to LoRAs/weights rebuilds them). "off" = always '
            'rebuild from scratch.'
        ),
    )

    opts.add_option(
        "fc_do_interrupt",
        OptionInfo(
            True,
            "Interrupt on Error",
            **args,
        )
        .info('if disabled, Forge Couple will simply "fail silently"')
        .needs_restart(),
    )

    opts.add_option(
        "fc_no_presets",
        OptionInfo(
            False,
            "Disable the Presets feature in Advanced mode",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_no_tile",
        OptionInfo(
            False,
            "Disable the Tile mode in img2img",
            **args,
        ).needs_reload_ui(),
    )

    opts.add_option(
        "fc_adv_newline",
        OptionInfo(
            False,
            "Keep newline characters in Advanced mode dataframe",
            **args,
        ).info('newlines would be shown as "\\n" literals'),
    )


on_ui_settings(fc_settings)
