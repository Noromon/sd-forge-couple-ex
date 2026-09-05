# SD Forge Attention Couple
This is an Extension for the Forge Webui, which allows you to ~~generate couples~~ target different conditionings at specific regions. No more color bleeds or mixed features!

> [!NOTE]
> - Ideas by human, implementation by LLM (*within 48h* so it might not be that elegant)
> - Experimental in nature; provided **as-is** without any technical support or maintenance commitment
> - Only limited real-world testing has been done on **Anima**

> Support [Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic) / [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

## Showcase
> Trying to generate "jesus christ arguing with santa claus"

<table>
  <thead align="center">
    <tr>
      <th><b>with</b> Forge Couple</th>
      <th><b>without</b> Forge Couple</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>
        <img src="example/on.jpg" width="384" /><br />Distinct and separate characters
      </td>
      <td>
        <img src="example/off.jpg" width="384" /><br />Features mixed between characters
      </td>
    </tr>
  </tbody>
</table>

## How to Use

> [!IMPORTANT]
> Only **SD1** and **SDXL** are supported<br>
> 🔥 **New:** Now supports **Anima**

> [!CAUTION]
> The effectiveness of this Extension depends on how well the Checkpoint follows the prompts<br>
> If the Checkpoint does not understand the composition to begin with, it will not generate the desired result accurately

> [!TIP]
> As shown in the various examples, it is recommended to still prompt for the total amount of subjects first within every region

<details open>
<summary><h4>Index</h4></summary>

- [Basic Mode](#basic-mode)
    - [Tile Direction](#tile-direction)
- [Advanced Mode](#advanced-mode)
- [Mask Mode](#mask-mode)
- [Separation Modes](#separation-modes)
- Misc.
    - [Global Effect](#global-effect)
    - [Compatibility](#compatibility-toggle)
    - [Separator](#couple-separator)
    - [Common Prompts](#common-prompts)
    - [LoRA](#lora-support)
- [Tile Mode](#tile-mode)
- [API](https://github.com/Haoming02/sd-forge-couple/wiki/API)
- [FAQ](#troubleshooting)

</details>

<br><hr><br>

### Basic Mode

The **Basic** mode works by dividing the image into multiple "tiles" where each tile corresponds to a "**[line](#couple-separator)**" of the positive prompt. Simply prompt more lines to configure more regions.

#### Tile Direction

For the **Basic** mode, you can choose to divide the image into columns or rows:

- **Horizontal:** Each line maps to regions from `left` to `right`
- **Vertical:** Each line maps to regions from `top` to `bottom`

<p align="center">
<img src="example/basic.jpg" width=512><br>
<code>Horizontal</code> <b>Direction</b>
</p>

```
2girls, blonde twintails, cyan eyes, white serafuku, standing, waving, looking at viewer, smile
2girls, black long hair, red eyes, dark school uniform, standing, crossed arms, looking away
```

<p align="center">
<img src="example/direction.jpg" height=512><br>
<code>Vertical</code> <b>Direction</b>
</p>

```
galaxy, stars, milky way
blue sky, clouds
sunrise, lens flare
ocean, waves
beach, sand
```

<br>

> [!NOTE]
> Most use cases can be achieved simply with the **Basic** mode... Stop overcomplicating things for no reason...

<br><hr><br>

### Advanced Mode

The **Advanced** mode allows you to manually specify the coordinates and sizes of each region

> [!IMPORTANT]
> The entire image **must** contain weights

- **Entries:**
    - Each row contains a range for **x** axis, a range for **y** axis, a **weight**, as well as the corresponding **line** of prompt
        - **x** axis is from `left` to `right`
        - **y** axis is from `top` to `bottom`
    - The range should be within `0.0` ~ `1.0`, representing the **percentage** of the full width/height *(**e.g.** `0.0` to `1.0` would span across the entire axis)*
    - The weight is capped between `0.0` ~ `5.0`
    - *(the actual generation is still based on the main prompt field, in case it went out of sync)*

> [!TIP]
> The mappings are not sent when using the `Send to img2img` function, click the `Pull from txt2img` to manually transfer the mappings

- **Control:**
    - Click on a row to select it, highlighting its bounding box
        - Click on the same row again to deselect it
    - When a row is selected, click the `🆕` button above / below to insert a new row above / below
        - If holding `Shift`, it will also insert a new empty line to the prompts
    - When a row is selected, click the `❌` button to delete it
        - If holding `Shift`, it will also **delete** the corresponding line of prompt
    - Click the `Default Mapping` button to reset the mappings

- **Presets:**
    - You can save the current mapping data, and load them again in the future

- **Draggable Region:**
    - When a bounding box is highlighted, simply drag the box around to reposition the region; drag the edges / corners to resize the region

- **Background:**
    - Click the `📂` button to load an image as the background for reference
    - Click the `⏏` button to load the **img2img** input image as the background
    - Click the `🗑` button to clear the background

<p align="center">
<img src="example/adv_ui.jpg" width=512><br>
<b>Advanced</b> Mode UI
</p>

<p align="center">
<img src="example/adv_result.jpg" width=512><br>
<b>Advanced</b> Mode Result
</p>

```
a cinematic photo of a couple, from side, outdoors
couple photo, man, black tuxedo
couple photo, woman, white dress
wedding photo, holding flower bouquet together
sunset, golden hour, lens flare
```

<br><hr><br>

### Mask Mode

The **Mask** mode allows you to manually draw each region

> [!IMPORTANT]
> The entire image **must** contain weights

- **Canvas:**
    - Click the **Create Empty Canvas** button to generate a blank canvas to draw on
    - Only **pure white** `(255, 255, 255)` pixels count towards the mask, other colors are simply discarded
        - This also means you can use other colors as the "eraser"
    - Click the **Save Mask** button to save the image as a new layer of masks
    - When a layer is selected:
        - Click **Load Mask** to load the mask into canvas
        - Click **Override Mask** to save the image and override the selected layer of mask
    - Click the **Reset All Masks** button to clear all the data

> [!TIP]
> The masks are not sent when using the `Send to img2img` function, click the `Pull from txt2img` to manually transfer the masks *(the `weights` are not sent...)*

- **Entries:**
    - Each row contains a **preview** of the layer, the corresponding **line** of prompt, and the **weight** for the layer
        - *(the actual generation is still based on the main prompt field, in case it went out of sync)*
    - Click on the preview image to <b>select</b> the layer
    - Click the arrow buttons to re-order the layers
    - Click the `❌` button to delete the layer

- **Uploads:**
    - Use the `Upload Background` to upload an image as reference to draw masks on
        - The image will be darkened, thus **not** counting towards the mask
    - Use the `Upload Mask` to upload an image as a mask that can directly be saved

> [!NOTE]
> The regions are not pixel-perfect, so just draw general shapes

> [!WARNING]
> For **Forge Classic** (`Gradio 3`) users, manually upload or simply drag & drop the images. Using `Ctrl + V` might send the image to the Canvas and break the Extension...

<p align="center">
<img src="example/mask_ui.jpg" width=512><br>
<b>Mask</b> Mode UI
</p>

<p align="center">
<img src="example/mask_result.jpg" width=512><br>
<b>Mask</b> Mode Result
</p>

```
cinematic photo of a dungeon
lit candles hanging on the wall
treasure chest
```

<br><hr><br>

### Separation Modes

When different regions use different LoRAs or styles, their features tend to bleed across region boundaries. The **Separation Method** control lets you choose how strictly each region is isolated:

| Mode | Isolation | Seam Coherence | Cost | Best For |
| :-- | :-- | :-- | :-- | :-- |
| `Attention` *(original)* | Low | High | 1x | Light separation, similar LoRAs |
| **`Hybrid`** *(recommended)* | Full | High | ~1x | Different LoRAs per region without hard seams |
| `Independent` | Strongest | Low | n x | Experimental: absolute isolation |
| `Latent` | Medium | Medium | 1x | Slightly stronger than Attention |

#### Attention *(Original)*

A single shared forward pass where all regions attend to the same k/v, with per-region attention outputs mask-blended afterwards. Lightest and fastest; works well when the LoRAs are not very different from each other. The **Mask Sharpening** control (`Soft` / `Hard` / adjustable `Temperature`) applies here to make the blending sharper.

#### Hybrid *(Recommended)*

Hybrid keeps **self-attention global** while routing **cross-attention and MLP per region**:

1. **Self-attention stays global with base weights** - every spatial token sees all others simultaneously, so composition, pose, and proportion remain coherent across regions from the first step to the last. This is what removes the "dimension wall" at seams that fully-independent approaches produce.
2. **Cross-attention & MLP are routed per region** - each token uses its own region's LoRA-patched sub-modules (and its own text context, with TE LoRAs isolated too), so features never bleed across regions despite sharing a single pass.

Cost is ~1x steps *(vs n x for Independent)*. The **Boundary Transition** control decides how tokens are assigned at the border: `Hard` gives each pixel to exactly one region; `Soft` blends a Gaussian transition band whose size and mix you tune with **Transition Width (px)** and **Soft Strength**.

#### Independent *(Experimental)*

A proof-of-concept for *absolute* LoRA isolation: n full inference passes, one per region, each running on base model + only that region's LoRAs. Isolation is total - but regions never see each other at all, which produces hard seams and costs n x time. Recommended for experimental use; prefer **Hybrid** for actual generation. Per-region `<lora:...>` tags in the prompt are parsed & applied independently, and the **Region Blend** control (`Hard` / `Feather`) merges the per-region latents.

#### Latent

The earliest separation attempt: instead of one shared forward, each region runs its own attention against its own k/v and the outputs are mask-blended - stronger isolation than `Attention`, still a single pass.

#### Comparison Showcase

Real-world comparisons generated with **Anima** - each pair uses a different character LoRA per region. The single-character references show what each character is supposed to look like; compare how well each mode keeps them distinct and coherent at the seam.

> [!NOTE]
> The **Low LoRA Weight** rows use lowered LoRA weights *(style `1.0` → `0.6`, characters `0.7` → `0.5`)* to test better compatibility under the `Attention` / `Latent` modes

##### Comparison 1 - Sleeping in a Train Carriage (Luo Tianyi (Mangzhong) & Hatsune Miku (Shaohua))

<p align="center">
<img src="example/sep_compare_1/char_left.jpg" width="256"><br>
<b>Luo Tianyi (Mangzhong)</b>
&nbsp;&nbsp;&nbsp;
<img src="example/sep_compare_1/char_right.jpg" width="256"><br>
<b>Hatsune Miku (Shaohua)</b>
</p>

| | Attention | Latent | Hybrid | Independent |
| :-- | :--: | :--: | :--: | :--: |
| **Normal Weights** | <img src="example/sep_compare_1/attention.jpg" width=384> | <img src="example/sep_compare_1/latent.jpg" width=384> | <img src="example/sep_compare_1/hybrid.jpg" width=384> | <img src="example/sep_compare_1/independent.jpg" width=384> |
| **Low LoRA Weight** | <img src="example/sep_compare_1/attention_low.jpg" width=384> | <img src="example/sep_compare_1/latent_low.jpg" width=384> | — | — |

<details>
<summary>Prompt</summary>

```
{comm: 
(masterpiece), best quality, score 9,
(2girls :1.4), (side by side), 
(@houkisei :0.2), (@suzumi narumi :0.2), (@alpha \(yukai na nakamatachi\) :0.2),
<lora:style-a:1>,  (loranlchstyle :1),
}, 
<lora:char-b:0.7>, (ltymz :1), (hair-ltymz), (hairflower-ltymz), (chinadress-ltymz :1), (armlet-ltymz :1), (highheels-ltymz :1),
low twintails, (luo tianyi :0.6),
china dress, pelvic curtain,
(shiny skin), (glistening skin :1.2), (glistening body), (light skin),
(medium breasts), 
(hands :1.2), nails,
{comm2: 
sleeping, (closed eyes), u u, 
(sitting), (sitting on seat), reclining, relaxed, (arms at sides :1.2), (hands down), 
(leaning to the side :1.4), (leaning on object), (leaning on), (leaning), 
(leaning on person), 
interior view, (interior), (subway train), (rapid transit :1.2), (train interior :1.4), seat, (window),
handbag, briefcase,
(overpass :1.2), bridge,
lake, (city), (sunset), cloudy, (orange sky), dusk, dusk shine, city horizon,
shiny, (backlighting :1.3), (raythalosm :1.3), (lens flare), (dramatic lighting :1.3), (detailed lighting :1.2), (ambient lighting :1.2),
}, 
BREAK
{comm}, 
<lora:char-c:0.7>, (mikush :1), (hair-mikush), chinadress-mikush, brooch-mikush, (shoes-mikush :1),
(light green hair :1), short sleeves, bare legs, china dress, (pelvic curtain :1),
(white socks), black shoes, 
(shiny skin :1.2), (glistening body :1.2), (glistening skin), (light skin),
medium breasts, 
hand, (cyan nails :1.1),
{comm2},
```

</details>

##### Comparison 2 - Selfie (Stardust V4 & XinHua AI)

<p align="center">
<img src="example/sep_compare_2/char_left.jpg" width="256"><br>
<b>Stardust V4</b>
&nbsp;&nbsp;&nbsp;
<img src="example/sep_compare_2/char_right.jpg" width="256"><br>
<b>Xinhua AI</b>
</p>

| | Attention | Latent | Hybrid | Independent |
| :-- | :--: | :--: | :--: | :--: |
| **Normal Weights** | <img src="example/sep_compare_2/attention.jpg" width=384> | <img src="example/sep_compare_2/latent.jpg" width=384> | <img src="example/sep_compare_2/hybrid.jpg" width=384> | <img src="example/sep_compare_2/independent.jpg" width=384> |
| **Low LoRA Weight** | <img src="example/sep_compare_2/attention_low.jpg" width=384> | <img src="example/sep_compare_2/latent_low.jpg" width=384> | — | — |

<details>
<summary>Prompt</summary>

```
{comm:
(masterpiece), best quality, score 9,
(2girls), (side by side),
(2girls taking selfie side by side, the left one is stardustv4, the right one is xinhuaai),
(@houkisei :0.2), (@suzumi narumi :0.2), (@alpha \(yukai na nakamatachi\) :0.2),
<lora:style-d:1>, (lorameionstyle :1),
}
<lora:char-e:0.7>, stardustv4, hair-stardustv4, costume-stardustv4, (thighboots-stardustv4 :1),
yellow eyes, light purple hair, (quad tails :1.3), ahoge, (long locks),
(bare shoulders), (halterneck), (halter dress), (two-sided dress),
(elbow gloves), (fingerless gloves),
(thigh high boots), (high heels),
(shiny skin :1.2), (light skin), glistening skin, glistening body,
(medium breasts), (cleavage),
hands, purple nails,
(blushing :1.2), (heavy breathing :1.3), (sweat :1.5), open mouth, lidded eyes,
(looking up), (looking at viewer),
(yuri :1.3), (symmetry :1.3),
light smile, (standing), (hands :1), asymmetrical docking, (selfie :1.2),
{comm2:
(full body :1.4), (duo focus),
(simple background :1.3), (white background :1.3),
shiny, (lighting :1.3), (raythalosm :1.3), (lens flare), (dramatic lighting :1.3), (detailed lighting :1.2), (ambient lighting :1.2),
}
BREAK
{comm},
<lora:char-f:0.7>,  (xinhuaai :1), (hair-xinhuaai :1.2), (costume-xinhuaai :1.2), (footwear-xinhuaai :1),
(long hair), (multicolored hair), pink eyes, hairpin, pink beret,
microskirt, sleeveless shirt, uneven gloves,
thighhighs, leather shoes,
(asymmetrical gloves), single elbow glove, (fingerless gloves),
(shiny skin :1.2), (light skin :1.2), glistening body, glistening skin,
big breasts, 
hands, pink nails,
lidded eyes, open mouth, evil smile,
(yuri :1.3), (symmetry :1.3),
light smile, (standing),
(looking up), (looking at viewer),
asymmetrical docking, (selfie :1.2),
{comm2},
```

</details>

##### Comparison 3 - Heart Hands (Luo Tianyi V4J & V4C)

<p align="center">
<img src="example/sep_compare_3/char_left.jpg" width="256"><br>
<b>Luo Tianyi V4J (JP ver.)</b>
&nbsp;&nbsp;&nbsp;
<img src="example/sep_compare_3/char_right.jpg" width="256"><br>
<b>Luo Tianyi V4C (CN ver.)</b>
</p>

| | Attention | Latent | Hybrid | Independent |
| :-- | :--: | :--: | :--: | :--: |
| **Normal Weights** | <img src="example/sep_compare_3/attention.jpg" width=384> | <img src="example/sep_compare_3/latent.jpg" width=384> | <img src="example/sep_compare_3/hybrid.jpg" width=384> | <img src="example/sep_compare_3/independent.jpg" width=384> |
| **Low LoRA Weight** | <img src="example/sep_compare_3/attention_low.jpg" width=384> | <img src="example/sep_compare_3/latent_low.jpg" width=384> | — | — |

<details>
<summary>Prompt</summary>

```
{comm: 
(masterpiece), best quality, score 9,
(2girls :1.4), (side by side), 
(@houkisei :0.2), (@suzumi narumi :0.2), (@alpha \(yukai na nakamatachi\) :0.2),
<lora:style-g:1>,  (loramacastyle :1),
}, 
<lora:char-h:0.7>, (ltyv4j :1), (hair-ltyv4j :1), (costume-ltyv4j :1), (footwear-ltyv4j :1),
(luo tianyi :1),
(grey hair:1.2), (green eyes :1.2), (short hair with long locks :1.2), (low twintails :0.8), 
off-shoulder shirt, crop top, yellow shorts,
midriff, stomach, (linea alba :1.2),
aqua thighhighs, high heels,
(shiny skin :1.2),
big breasts, 
hands, blue nails,
{comm2: 
(yuri :1.3), (symmetry :1.3),
light smile, (standing), asymmetrical docking, (symmetrical hand pose :1.4), 
(heart hands duo :1.3), (heart hands :1.4),
(full body :1.4), (duo focus), (high angle :1.2), (foreshortening), 
(indoors), (bedroom), shiny, (sidelighting :1.3), (raythalosm :1.3), (lens flare), (dramatic lighting :1.3), (detailed lighting :1.2), (ambient lighting :1.2),
}, 
BREAK
{comm}, 
<lora:char-i:0.7>,  (ltyv4c :1), (hair-ltyv4c :1), (headset-ltyv4c :1), (dress-ltyv4c :1), (necktie-ltyv4c :1), (sleeves-ltyv4c :1), (bracelet-ltyv4c :1), (socks-ltyv4c :1), (boots-ltyv4c :1),
(grey hair), (green eyes :1.2), (short hair with long locks :1), (luo tianyi :1),
microskirt,
single thighhigh, single kneehigh, 
(asymmetrical footwear),
(shiny skin :1.2), (light skin :1.2), glistening body, glistening skin,
big breasts,
(hands), blue nails,
{comm2},
```

</details>

### Global Effect

In **Basic** and **Mask** modes, you can set either the **first** line or the **last** line of the positive prompt as the "background," affecting the entire image, useful for styles or quality tags.

### Compatibility Toggle

When this is enabled, the Extension will not function during the `Hires. Fix` pass.
*Personal suggestion: Turn this thing **OFF** while using multiple LoRAs.*

### Couple Separator

By default when the field is left empty, this Extension uses the newline character (`\n`) as the separator to determine "lines" of the prompts. You may also specify other words as the separator instead.

> [!TIP]
> To keep the custom separator within its own line, you can add `\n` before and after the word

- <ins>Examples</ins>
    - ` ` *(left empty)*
    - `foo`
    - `\nbar\n`
    - `\n\n`

### Common Prompts

If you have multiple characters that share the same outfits, poses, or expressions, you can now simplify the process via **Common Prompts** - No more copying and pasting the same lines over and over!

0. Select a syntax between `{ }` or `< >`
1. Define an unique key *(**e.g.** `cloth`)*
2. Follow up with a `:` *(**e.g.** `cloth:`)*
3. Follow up with your common prompts *(**e.g.** `cloth:t-shirt, jacket, jeans`)*
4. Surround the whole thing with your chosen brackets *(**e.g.** `{cloth:t-shirt, jacket, jeans}`)*
5. Finally, you can now use the key to recall the common prompts in other lines *(**i.e.** `{cloth}`)*

- **TL;DR:** If you have `{foo:bar}` in your prompt, every occurrence of `{foo}` *(and the original `{foo:bar}`)* will be replaced with `bar` during the generation
    - You can also omit the original `{foo:bar}` by disabling the `Include Definitions in Prompt` option

> [!IMPORTANT]
> - The key has to be unique
> - You can have more than multiple common prompts at the same time
> - Each bracket can only contain one key

> [!TIP]
> You can enable **Debug** to check if it is working as intended in the console

<p align="center">
<img src="example/common.jpg" width=512><br>
<b>Anima</b> X <code>Common Prompt</code>
</p>

<details>
<summary>Infotext</summary>

```
masterpiece, best quality, good quality, absurdres, newest. 3girls standing side-by-side, each holding a sign.
3girls, hatsune miku, {common:vocaloid, casual, clothed, looking at viewer, smile}, holding a sign that says "Forge".
3girls, kagamine rin, {common}, holding a sign that says "Couple".
3girls, kasane teto, {common}, holding a sign that says "Anima".
Negative prompt: monochrome, greyscale, loli, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, watermark, worst quality, low quality, large breasts, muscular, deformed hands, bad anatomy, extra limbs, poorly drawn face, mutated, extra eyes, bad proportions, character doll, chibi, old, early, censored, 3d, high contrast, ai-generated
Steps: 32, Sampler: Euler a, Schedule type: Normal, CFG scale: 5, Shift: 3, Seed: 2984220975, Size: 1344x1024, Model hash: 14fffe8ad5, Model: anima-preview3-base, Clip skip: 2, RNG: CPU, forge_couple: True, forge_couple_compatibility: True, forge_couple_mode: Basic, forge_couple_separator: \n, forge_couple_direction: Horizontal, forge_couple_background: First Line, forge_couple_background_weight: 0.5, forge_couple_common_parser: { }, forge_couple_def_in_prompt: True, Version: neo, Module 1: qwen_3_06b, Module 2: qwen_image_vae
```

</details>

### LoRA Support

LoRA that contains multiple subjects is easier to generate multiple characters. Using different LoRAs in different regions depends on how well the LoRAs' concepts work together...

<br><hr><br>

## Tile Mode

The **Tile** mode allows you to upscale an image while keeping the features separated by prompting each tile based on its overlapping regions.

- **Prerequisite**
    - A way to break the generation into tiles
        - **e.g.** the built-in `SD Upscale` script
        - The tile order has to be `top-left` > `top-right` > `bottom-left` > `bottom-right`
    - The input image does **not** need to have been generated with `ForgeCouple`
    - Set up the `ForgeCouple` regions like you would normally *(all 3 modes are supported)*
    - Switch to the `Tile` tab and enable the feature
    - `ControlNet` with the `Tile` Module is recommended

- **Parameters**
    - **Inclusion Threshold** controls how much overlap between the tile and the region is needed for the corresponding prompt to be included
        - Prevents adding the prompts from regions barely touching the tile
        - `0.0` means every single line of prompt would be included; `1.0` means the region and the tile have to perfectly match to be counted
    - Set the **Scale Factor** and **Tile Overlap** to match the **SD Upscale** settings; click **Calculate**; it should automatically populate the correct **Final Width/Height** and **Column/Row Counts**
    - **Subject Replacement** is used to replace the original "total amount of subjects" with singular subject, in order to prevent generating extra characters within a tile
        - Each line is a `key: values` pair
        - The `key` is the prompt to use
        - The `values` are the tags to be replaced, separated by comma
        - *(Alternatively, you can just modify the original prompts and ignore this field)*
    - You may enable **Debug Tiles** to check if the prompts are assigned correctly

<p align="center">
<img src="example/tile_ui.jpg" width=512><br>
<b>Tile</b> Mode UI
</p>

> [!Tip]
> Both **Global Effect** and **Common Prompts** also work with `Tile` mode

<br>

## API
For usages with API, please refer to the [Wiki](https://github.com/Haoming02/sd-forge-couple/wiki/API)

<br><hr><br>

## Troubleshooting
> **F**requently **A**sked **Q**uestions

- **Generation gets interrupted at 1st step**
    - In `Forge`, when raising an Error from an Extension, it only gets caught while the generation continues, leading to `ForgeCouple` failing "silently." To work around this, `ForgeCouple` now interrupts the generation when an error occurs. Check the Console logs to see what went wrong...
    - *(you could disable this behavior in the settings)*

- **Not Enough Lines in Prompt**
    - In **Basic** mode, you need at least **2** lines of prompts for it to tile; **3** in case you enable **Global Effect**

- **Number of Couples and Masks mismatched**
    - Similarly, the number of lines in prompts should match the number of regions defined in **Advanced** and **Mask** modes

> [!IMPORTANT]
> Empty lines are still counted; ensure you do not leave an empty line at the end; if you want to have an empty line between each region for clarity, adjust the **Couple Separator**

- **Image must contain weights on all pixels**
    - As mentioned in [Advanced](#advanced-mode) and [Mask](#mask-mode), the entire image must contain weights. This error occurs when you didn't fill the whole image. The easiest way to achieve this:
        - **Advanced:** Create a layer that covers the entire image *(**i.e.** `0.0, 1.0, 0.0, 1.0`)*
        - **Mask:** Use the **Global Effect**

- **Incompatible Extension**
    - Certain Extensions, such as [sd-dynamic-prompts](https://github.com/adieyal/sd-dynamic-prompts), will also process the prompts before/during generation. These may break the **Couple Separator** and/or **Common Prompts** as a result.

- **TypeError: 'NoneType'**
    - For users that get the following error:

    ```py
    RuntimeError: shape '[X, Y, 1]' is invalid for input of size Z
    shape '[X, Y, 1]' is invalid for input of size Z
    *** Error completing request
        ...
        Traceback (most recent call last):
            ...
            res = list(func(*args, **kwargs))
        TypeError: 'NoneType' object is not iterable
    ```

1. Go to **Settings** > **Optimizations**, and enable `Pad prompt/negative prompt`
2. Set the `Width` and `Height` to multiple of **64**

<br><hr><br>

## Special Thanks
- Credits to the original author, **[laksjdjf](https://github.com/laksjdjf)**, whose [ComfyUI Node](https://github.com/laksjdjf/cgem156-ComfyUI/tree/main/scripts/attention_couple) I referenced to port into Forge

<pre align="center">
Copyright (C) 2023 laksjdjf
Copyright (C) 2026 Haoming02
Copyright (C) 2026 Noromon

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
</pre>
