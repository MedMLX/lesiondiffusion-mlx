# lesiondiffusion-mlx

Add synthetic lesions to CT with native MLX, with controls for organ, size, placement and phenotype.

[Docs](https://github.com/MedMLX/lesiondiffusion-mlx/blob/main/docs/technical.md) · [Model weights](https://huggingface.co/MedMLX/lesiondiffusion-mlx)

| Input | Output |
| :--- | :--- |
| CT + aligned binary organ mask | Synthetic CT and lesion mask |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "lesiondiffusion-mlx>=0.2.5"
```

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

Weights download from `MedMLX/lesiondiffusion-mlx` on first use. A local native
folder can also be passed as `weights_path`; it needs `model.safetensors` and
`tokenizer.json`. Conversion receipts are not required.

Use `lesion_mask_path` instead of `bbox_size_mm` to inpaint a supplied lesion shape. Inputs must support the documented crop sizes. Outputs use a RAS 1 mm grid; voxels outside the lesion can change. Synthetic output is not ground truth.

## License

Apache-2.0; license and third-party notices are included in the distribution. Model weights retain their publisher terms.

For research and education. Outputs require human review.
