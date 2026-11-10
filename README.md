<div align="center">

# lesiondiffusion-mlx

Add synthetic lesions to CT with native MLX, with controls for organ, size, placement and phenotype.

[![PyPI](https://img.shields.io/pypi/v/lesiondiffusion-mlx)](https://pypi.org/project/lesiondiffusion-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/lesiondiffusion-mlx)](https://pypi.org/project/lesiondiffusion-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/lesiondiffusion-mlx/blob/v0.2.8/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/lesiondiffusion-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/lesiondiffusion-mlx/main/docs/showcase.png" alt="Liver, kidney, lung and bladder CT with the target site boxed, and a close-up of each generated lesion" width="100%">

<sub>Source CT from [KiTS23](https://kits-challenge.org/kits23/), CC BY-NC-SA 4.0, and [LUNA16](https://zenodo.org/records/3723295), CC BY 4.0.</sub>

</div>

| Input | Output |
| :--- | :--- |
| CT + aligned binary organ mask | Synthetic CT and lesion mask |

| Model | Weights download | Peak MLX memory | Measured on |
| :--- | ---: | ---: | :--- |
| `lesiondiffusion`, conditioned mode (default) | 1.7 GB | 5.19 GB | 160×160×128, 1 mm RAS crop, 200 diffusion steps |
| `lesiondiffusion`, source mode | 1.7 GB | 3.46 GB | 160×160×128, 1 mm RAS crop, 200 diffusion steps |

Requires Apple Silicon with Metal. The [technical notes](https://github.com/MedMLX/lesiondiffusion-mlx/blob/main/docs/technical.md)
give the precision of each network in these measurements.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "lesiondiffusion-mlx>=0.2.8"
```

## Run

```python
from lesiondiffusion_mlx.medmlx import run

result = run(
    image_path="ct_1mm_ras.nii.gz",
    organ_mask_path="liver_mask.nii.gz",
    output_dir="out",
    organ="liver",
    lesion_type="tumor",
    bbox_size_mm=[30, 30, 20],
    placement="inside_parenchyma",
)
print(result["outputs"]["synthetic_image"])
```

Weights download from `MedMLX/lesiondiffusion-mlx` on first use; `medmlx setup lesiondiffusion`
downloads them ahead of time. To use your own copy, pass a folder containing
`model.safetensors` and `tokenizer.json` as `weights_path`. Conversion receipts are not
required.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One 3D CT in NIfTI (`.nii` or `.nii.gz`), in Hounsfield units |
| `organ_mask_path` | Binary organ mask (background 0, organ 1) on the CT grid |
| `output_dir` | Directory for the outputs; created if missing |
| `organ` | A key of `lesiondiffusion_mlx.medmlx.SOURCE_ORGAN_LABELS`, such as `liver` |
| `lesion_type` | Free text such as `"tumor"` |
| `bbox_size_mm` | Lesion box extent (X, Y, Z) in mm; required unless `lesion_mask_path` is given |
| `placement` | `inside_parenchyma`, `protruding_from_parenchyma`, `within_lumen` or `wall_thickening` |
| `lesion_mask_path` | Optional lesion shape (background 0, lesion 1) to inpaint instead of generating one |
| `anatomy_mask_path` | Optional 104-label surrounding-organ map that conditions mask generation |
| `bbox_center_voxel` | Optional `[I, J, K]` box center on the standardized grid; placement is automatic otherwise |
| `diffusion_steps`, `seed` | Sampling controls; the upstream defaults are 200 and 23 |
| `inference_mode` | `"conditioned"` (default) or `"source"`; see below |
| `output_region` | `"full"` (default), `"box"` or `"lesion"`; see below |
| `weights_path` | Optional folder containing `model.safetensors` and `tokenizer.json` |
| `overwrite` | `False` by default; existing outputs are never replaced |

**Phenotype text.** `shape`, `density`, `density_variation`, `enhancement`, `surface`,
`adjacent_relationship` and `specific_features` default to the upstream wording
(irregular, hypodense, heterogeneous, non-contrast CT, ill-defined margin). They
condition the generator but do not guarantee a particular HU contrast or appearance.

**Input geometry**

- The CT and masks must be finite, axis-aligned grids that match after they are
  reoriented and resampled to RAS 1 mm; oblique or sheared grids are refused.
- They must fit the 64×152×152 mask crop and the 128×128×128 inpainting crop (Z, Y, X);
  with `lesion_mask_path` only the inpainting crop applies.
- A supplied lesion mask must be a single connected component that overlaps the organ
  mask, so split multifocal segmentations first.
- The box guides generation, but generated lesions can extend past the box or organ.

**Outputs.** `synthetic_ct.nii.gz` (float32 HU), `lesion_mask.nii.gz` and
`organ_mask.nii.gz` (uint8) on the standardized grid. The decoder reconstructs the whole
CT, and that reconstruction does not reproduce the input exactly: away from the lesion
it shifts attenuation by several HU, by an amount that depends on the volume.
`output_region` controls how much of it reaches `synthetic_ct.nii.gz`:

| `output_region` | `synthetic_ct.nii.gz` holds |
| :--- | :--- |
| `"full"` (default) | The reconstruction, as upstream writes it, so every voxel can change |
| `"box"` | The reconstruction over the lesion box, feathered into the input over 6 mm. Tissue inside the box keeps the shift |
| `"lesion"` | Also decodes the unchanged input, and adds only the difference between the two decodes to the input CT, within 33 voxels of the lesion along each axis. The shift cancels, and voxels farther away are the input unchanged. Use it when the surrounding anatomy should keep its measured values |

`"box"` and `"lesion"` also write the full reconstruction as `raw_synthetic_ct.nii.gz`.
`run` returns `{"outputs": {...}, "details": {...}}`, with run metadata in `details`.
Invalid inputs raise `InvalidInputError`; missing or mismatched weights raise
`AssetNotReadyError`. Synthetic output is not ground truth.

### Inference modes

| `inference_mode` | Behaviour |
| :--- | :--- |
| `"conditioned"` (default) | Enables the inpainting network's trained text context, which the pinned upstream configuration turns off, and runs FP16 convolution, fused attention and compiled denoisers natively. Decodes in FP32, because the decoder's activations can exceed the FP16 range on some volumes |
| `"source"` | Follows the upstream inference configuration and arithmetic for reproduction, including FP16 decoding; when that overflows, the run raises an error instead of writing a CT. Results can differ from conditioned output |

MLX and Torch random streams differ, so a seed repeats results only within MLX.
Inference requires Apple Silicon with Metal.

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `lesiondiffusion` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[generate,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe lesiondiffusion`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## License

[Apache-2.0](https://github.com/MedMLX/lesiondiffusion-mlx/blob/v0.2.8/LICENSE). See [third-party notices](https://github.com/MedMLX/lesiondiffusion-mlx/blob/v0.2.8/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.
