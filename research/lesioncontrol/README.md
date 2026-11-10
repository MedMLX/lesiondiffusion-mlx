# LesionControl

Control 3D lesion appearance with a reference tumor and measured radiomic targets.
LesionControl uses LesionDiffusion for MLX generation and mlx-radiomics for the
measurement feedback. It fits each requested reference profile rather than
applying one appearance preset to every tumor.

The research prototype accepts prepared 128³, 1 mm RAS CT and mask inputs. Reference
and host are separate: generation uses a tumor-free host and a supplied placement
outline. Appearance targets can preserve absolute attenuation or contrast against
normal liver. Supplied shape features are excluded from appearance fitting.

Place the `lesiondiffusion-mlx`, `mlxradiomics` and `medmlx` checkouts beside each
other. From this directory (`research/lesioncontrol`), run `make env`. The project
creates its own pinned macOS arm64 environment and uses this parent runtime
checkout plus the sibling library sources. Supply model weights, adapter
checkpoints and image data separately.

```bash
.venv/bin/lesioncontrol profile \
  --image reference_ct.nii.gz --mask reference_tumor.nii.gz \
  --liver-mask reference_liver.nii.gz --target-space liver_contrast \
  --output reference.json

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
must already share a 128³, 1 mm RAS grid. LesionControl preserves that grid and
does not prepare or register scans.

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
