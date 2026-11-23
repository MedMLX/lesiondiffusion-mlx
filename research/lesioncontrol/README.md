# LesionControl

LesionControl uses LesionDiffusion and MLXRadiomics to generate a liver lesion
from measurements of a reference tumor. It profiles the reference on a 128³,
1 mm RAS CT, then runs diffusion on a separate tumor-free host. LMNet can
generate the contour, or LINet can inpaint a supplied outline. Appearance
targets use absolute attenuation or contrast against surrounding liver.

Place the `lesiondiffusion-mlx`, `mlxradiomics` and `medmlx` checkouts beside each
other. From this directory (`research/lesioncontrol`), run `make env`. The project
creates its own pinned macOS arm64 environment and uses this parent runtime
checkout plus the sibling library sources. Supply model weights, adapter
checkpoints and image data separately.

## Generate shape and appearance

Profile a reference tumor, then generate on a separate clean liver crop of at
least 152 × 152 × 128 voxels on a 1 mm RAS grid.

```bash
.venv/bin/lesioncontrol profile \
  --image reference_ct.nii.gz --mask reference_tumor.nii.gz \
  --liver-mask reference_liver.nii.gz --target-space liver_contrast \
  --spatial --output reference.json

.venv/bin/lesioncontrol synthesize \
  --profile reference.json --host clean_host.nii.gz --liver-mask host_liver.nii.gz \
  --known-tumor-mask host_known_tumors.nii.gz --weights weights_directory \
  --seed 23 --output results/spatial
```

The reference supplies box dimensions, shape measurements and attenuation
summaries. Reuse a profile on another patient's liver to generate a lesion with
that phenotype. LMNet proposes 12 contours at one site; `--candidates` sets
the number. LesionControl scales each contour along its principal axes to the
reference volume, elongation and flatness, and selects the closest complete
contour by sphericity, maximum diameter and solidity. LINet runs once, on
the selected contour, to supply the spatial texture. Calibration adjusts its
mean, rim-to-core trend, coarse variation and excessive sharpness at the liver
interface when the profile includes a reference liver mask. With that mask,
reference voxels within 2 mm of the liver surface do not contribute to the
texture targets.

With `liver_contrast` targets, the lesion keeps the host scan's own sub-2 mm
noise, scaled to the reference's lesion-to-liver ratio, so its grain follows
the host scan. Pass `--reference-noise` to use the reference's fine-texture HU
amplitude.

Automatic placement uses the requested size and available liver tissue. A
contour that crosses the liver surface by up to 1% of its voxels is trimmed
there. A contour that crosses further moves whole to the nearest fitting site.
Use `--center I J K` to fix the box center. With `--max-translation-mm 3`, the
workflow can move a contour up to 3 mm from a fixed center. Pass
`--placement-mask outline.nii.gz` to hold the contour fixed and run LINet once
on a crop of at least 128³ voxels.

Use `--match-liver-surface` with automatic placement to match how much of the
reference tumor boundary lies within 3 mm of the segmented liver surface.
The reference spatial profile must include a liver mask. The workflow chooses
the nearest shift within one percentage point of the target and trims up to
10% of the contour at the liver surface when a whole contour cannot reach it.
If no shift reaches the target, it uses the closest attainable fraction. The
achieved contact is recorded. You can limit the displacement with
`--max-translation-mm`.

The output directory contains the calibrated `synthetic_ct.nii.gz`,
`lesion_mask.nii.gz`, the original draws and `synthesis.json` with the selected
seed, translation in voxels, pre-calibration CT path, shape measurements and
all 101 appearance-feature differences.

## Match a supplied contour

```bash
.venv/bin/lesioncontrol match \
  --profile reference.json --host clean_host.nii.gz \
  --liver-mask host_liver.nii.gz --placement-mask outline.nii.gz \
  --known-tumor-mask host_known_tumors.nii.gz \
  --weights weights_directory --budget 12 --seed 23 --output results/match

.venv/bin/lesioncontrol generate \
  --policy results/match/policy.json --seed 24 --output results/seed24
```

`absolute_hu` preserves the reference's HU targets. `liver_contrast` measures
the reference relative to its liver ring and compares generated features after
subtracting the unchanged host's liver-ring mean. This allows host liver
attenuation to differ from the reference scan. Both use the same 25 HU feature
bins and six appearance families. Shape is fixed by the supplied outline.

The entire host input must be free of known tumor, including outside the
placement region. Supply its known-tumor mask or explicitly use `--declared-clean`
for an input whose tumor-free status was established separately. CT and masks
must already share a 1 mm RAS grid. The supplied-outline matching workflow uses
128³ crops.

The search varies retained DPM++ steps, density, homogeneity, margin description,
HCC description and numeric text conditioning. It saves a reusable policy,
generated images, feature differences and geometry checks. `profile --target
FEATURE=VALUE` can edit an individual target; mutually incompatible targets may
remain unmatched.

The controller reports selected settings and remaining feature differences.
Saved policies include unchanged-host and untuned-generation scores. A reference
QC JSON object can be passed with `profile --qc-record`; explicitly quarantined
appearance references are rejected.
Comparisons to a supplied reference profile measure conditional fidelity. The
learned adapter adds numerical radiomic conditioning before LINet's attention
cache. Adapter checkpoints are supplied separately and run on MLX Metal.

Use a trained checkpoint directly with a compatible profile:

```bash
.venv/bin/lesioncontrol sample \
  --profile reference.json --host clean_host.nii.gz \
  --liver-mask host_liver.nii.gz --placement-mask outline.nii.gz \
  --known-tumor-mask host_known_tumors.nii.gz \
  --weights weights_directory --adapter adapter_directory \
  --adapter-gain 1 --base-steps 25 --retained-steps 25 \
  --seed 24 --output results/learned
```

The checkpoint declares its feature space and training-only normalization.
`--adapter-gain 0` uses the same text schema and sampling controls with the learned
residual disabled, providing a matched comparison. Omitting `--adapter` uses the
native text schema.

See [technical notes](docs/technical.md) for the objective, numerical contracts
and learned-conditioning preflight. The [research entrypoints](docs/research.md)
describe the external input schemas, CUDA training environment and fixed-study
evaluation commands. Run `make qa` and `make test TESTS=tests` from this directory
for its source checks and synthetic contract tests; the parent runtime has its
own checks.

Research precedents include [Tumor Synthesis Conditioned on Radiomics](https://openaccess.thecvf.com/content/WACV2025/papers/Kim_Tumor_Synthesis_Conditioned_on_Radiomics_WACV_2025_paper.pdf)
and [RadiomicsFill-Mammo](https://papers.miccai.org/miccai-2024/paper/1807_paper.pdf).
