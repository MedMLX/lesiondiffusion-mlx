# lesiondiffusion-mlx technical notes

Full details behind the [README](../README.md).

## Overview

Native MLX implementation of [LesionDiffusion](https://github.com/HengruiTianSJTU/LesionDiffusion)
for text-conditioned CT lesion insertion. The reference is upstream revision
[`880736e2b8a4d9becdb6594da7f8af6db19d04d1`](https://github.com/HengruiTianSJTU/LesionDiffusion/tree/880736e2b8a4d9becdb6594da7f8af6db19d04d1),
including `InferMaskDiffusion` and the configured `InferInpaintingDiffusion_v2`.
The default `conditioned` inference mode enables LINet text context, compiles both
denoisers and uses native kernels: FP16 convolution (implicit GEMM on the GPU
neural accelerators of M5-class Macs, fused with the preceding group norm and the
following residual additions), fused attention and single-pass group
normalization. The explicit `source` mode retains the pinned inference
configuration and source arithmetic for reproduction.
BiomedCLIP's text pooler/projection follows OpenCLIP 2.24.0 at
[`3ff1faf10b60be27252be7f6c84ce7c8c5e14ec8`](https://github.com/mlfoundations/open_clip/tree/3ff1faf10b60be27252be7f6c84ce7c8c5e14ec8).
The canonical revisions and source files are in
[`reference.json`](../src/lesiondiffusion_mlx/reference.json).

LMNet, LINet, VQ and BERT learned layers execute in MLX. Shared runtime checks,
errors and array primitives come from `medmlx-core`. Importing this package does
not import MLX. Inference does not import Torch or MONAI and has no CPU execution
fallback or device selector.

## Installation and weights

Python 3.13 or 3.14 and an Apple Silicon Mac with accessible Metal are required for native
inference. Create environments, caches, reference sources and checkpoints outside
this checkout:

```sh
export UV_CACHE_DIR="$TMPDIR/uv-cache"
export RUFF_CACHE_DIR="$TMPDIR/lesion-ruff-cache"
export HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/lesion-hypothesis"
export PYTEST_ADDOPTS="-o cache_dir=$TMPDIR/lesion-pytest-cache"
uv venv "$TMPDIR/venv-lesiondiffusion-mlx" --python 3.13
uv pip install --python "$TMPDIR/venv-lesiondiffusion-mlx/bin/python" -e '.[dev,reference]'
```

MLX, which runs the networks, is installed as a dependency of `medmlx-core`.
Version 0.2.3 uses `medmlx-core==0.2.5` and downloads the native bundle from
`MedMLX/lesiondiffusion-mlx` when `weights_path` is omitted. An explicit folder
never triggers a download. It contains
`model.safetensors` and `tokenizer.json`. Safetensors keys are prefixed `lmnet.`,
`linet.`, `vq.`, `lmnet_text.` and `linet_text.`; source mode does not require
`linet_text`. Conversion receipts are not runtime inputs. Results record the
managed folder's downloaded commit as `details.weights_revision`; explicit
unmanaged folders report `None`.

## File runner

The `medmlx.models` entry point exposes `lesiondiffusion_mlx.medmlx:MODEL` and its
keyword-only `run` function:

```python
from lesiondiffusion_mlx.medmlx import run

result = run(
    image_path="ct_1mm_ras.nii.gz",
    organ_mask_path="liver_mask.nii.gz",
    output_dir="lesion-output",
    organ="liver",
    lesion_type="tumor",
    bbox_size_mm=[30, 30, 20],
    placement="inside_parenchyma",
    seed=4,
)
```

CT and binary organ mask (background 0, organ 1) must be finite and have matching
axis-aligned NIfTI grids after standardization, large enough to accommodate LMNet's
`(64,152,152)` and LINet's `(128,128,128)` crops (source Z,Y,X order); with a supplied
lesion mask only LINet's crop applies. Crop requirements are checked during geometry
preparation, before loading the networks.
The runner reorients, resamples and pads as needed. In `conditioned` mode, linear
CT interpolation and nearest-neighbor mask sampling use the physical coordinates
of the output affine. The reoriented origin is retained, output extents are
`round(size * spacing_mm)`, and the 1 mm grid uses edge extension for CT and zero
background for masks. Integer CT is interpolated in float32. `source` mode retains
upstream's `scipy.ndimage.zoom` convention from `preprocess.standardize_image`
(`pipeline/preprocess.py`), including its mismatch between interpolation coordinates
and the assigned 1 mm affine. The selected convention is recorded in
`generation_spec.resampling`; the geometry helpers default to `resampling="source"`
and accept `resampling="affine"` for the conditioned convention.
Label integrality, nonnegativity and allowed IDs are checked before resampling,
including voxels that downsampling would omit.
Already-RAS 1 mm even input is returned unchanged. A grid that is still not axis-aligned after
reorientation (a true oblique or sheared affine) is refused rather than resampled;
this port implements reorientation and isotropic resampling, not general affine
registration. `organ` accepts all 104 upstream anatomical labels, with names
lowercased and spaces replaced by underscores; see `SOURCE_ORGAN_LABELS` for exact
spellings.
This vocabulary does not establish model quality for every anatomy.
An optional `anatomy_mask_path` supplies the upstream 104-class surrounding-organ
map. LMNet receives this map divided by 104, the binary target organ and the box,
matching `InferMaskDiffusion.log_images` and `MaskDiffusion.get_input`. The map
uses the same nearest-neighbor standardization as the binary mask; unknown IDs,
grid differences and disagreement with the selected organ are refused. Omitting
it conditions LMNet on the target organ only.
`lesion_type` and phenotype fields are
free text, with no additional organ/lesion pairing whitelist.

`placement` is one of `inside_parenchyma`, `protruding_from_parenchyma`,
`within_lumen` or `wall_thickening`. Automatic placement uses upstream's MT19937
shuffle, hollow-organ erosion and boundary adjustment. An explicit
`bbox_center_voxel=[I,J,K]` and its box must fit the even-padded standardized RAS 1 mm
grid. Box dimensions are rounded to voxels and use upstream center ± floor(size/2)
slices, so odd voxel extents lose
one voxel. Placement guides the reference box; generated masks are selected by
upstream threshold/component/volume QC and may extend beyond the box or organ.

Version 0.2.3 accepts an optional `lesion_mask_path` to replace LMNet's
mask with a supplied lesion shape:

```python
result = run(
    image_path="ct_1mm_ras.nii.gz",
    organ_mask_path="liver_mask.nii.gz",
    lesion_mask_path="tumor_outline.nii.gz",
    output_dir="outline-output",
    organ="liver",
    lesion_type="tumor",
    placement="inside_parenchyma",
    output_region="box",
)
```

Upstream LINet inpaints whichever label file its list names and reads label 2 as
the lesion (`InpaintingDiffusion_v2.get_input`); LMNet writes that file with organ 1
and lesion 2. The runner hands LINet a boolean lesion, so the supplied file uses the
`target_mask` encoding: background 0, lesion 1. It passes the same label checks,
nearest-neighbor standardization and zero padding as the organ mask and must share
the CT grid. LMNet is not loaded. Upstream inpaints each connected component larger
than 50 voxels (up to three) in its own crop; this runner inpaints one lesion in one
crop, so the mask must be a single face-connected component after standardization.
Split multifocal segmentations first. The lesion must overlap the organ mask, whose
name is LINet's lesion-location text, but may extend past it; the overlap is
recorded. LINet sees the mask on its half-resolution latent, which samples even
source-order voxels, and samples only a `(128,128,128)` crop centered on the lesion.
A lesion with no voxel on that lattice (for example one voxel thick), or with
latent voxels outside the crop, is refused, because it would be labeled but not
painted.

The box is then the lesion's bounding box widened outward to even voxel indices,
which are whole latent cells. `generation_spec.bbox_center_voxel` and
`bbox_size_voxel` record it, and `output_region="box"` keeps it generated
(`compositing.kept_region="lesion_mask_box"`) with the usual 6 mm feather.
`bbox_size_mm` is required without a lesion mask; `bbox_size_mm`,
`bbox_center_voxel` and `anatomy_mask_path` (LMNet-only context) are rejected with
one. `placement` and the phenotype fields still form LINet's text in conditioned
mode; `placement` is not checked against the mask shape. `lesion_type` then only
names the output label, and `seed` seeds LINet sampling. `target_mask` is the
supplied mask on the standardized grid. Details record
`native_runtime.lesion_mask_source="supplied"`, `stages.lmnet.status="skipped"` and
`generation_spec.lesion_mask` (file, voxel counts and box rule). LINet paints a new
lesion with the supplied shape; it does not copy intensities from the scan the
outline came from. Inpainting quality for supplied masks, compared with LMNet masks,
is not qualified.

Sampling defaults follow upstream: `diffusion_steps=200` and `seed=23`.
Package text defaults are `shape="irregular"`,
`density="hypodense lesion"`, `density_variation="heterogeneous"`,
`enhancement="non-contrast CT"`, `surface="ill-defined margin"`,
`adjacent_relationship="no close relationship with surrounding organs"` and
`specific_features=""`. Steps must produce valid upstream uniform DDIM indices;
there is no separate 10–200 step restriction. Sampling uses upstream eta=1.
A seed repeats placement and native MLX sampling within MLX; Torch and MLX RNG
streams differ, so a shared integer seed does not imply shared noise.

`inference_mode="conditioned"` is the default. The released LINet checkpoint
contains learned context projection and cross-attention parameters, but the
pinned inference YAML sets `skip_context=True`, disabling those parameters and
the requested density/enhancement conditioning. Conditioned mode enables LINet
context and evaluates its FP32 BiomedCLIP encoder once per run. It passes the
phenotype attributes, excluding `size` and the automatically constructed
`CT value` sentence: that sentence used the input organ's mean HU, which is not a
requested lesion density. LMNet retains FP32 precision and excludes density from
mask conditioning.

Conditioned mode also uses native FP16 convolution for LINet and VQ. This
changes convolution accumulation and bias-rounding behavior, and enabling context
changes the denoiser itself. Sampled latents, VQ code selection and final CT can
differ from source mode; source-mode numerical bounds do not qualify conditioned
generation. Text requests reach the network but do not guarantee a particular
HU contrast or clinical appearance. There is no numeric mean-density correction.

Conditioned mode compiles each stage's bound denoiser with fixed loaded weights
and uses these kernels, which keep FP32 accumulation but not source operation
order:

- On Apple GPUs of generation 17 or newer (M5 family) with macOS 26.2 or newer,
  FP16 padding-1 3x3x3 convolutions in LINet and VQ (stride 1 or 2) run as an
  implicit GEMM on the GPU neural accelerators (`_nax_conv`). Each output
  accumulates all products in FP32, adds the bias in FP32 and rounds to FP16
  once; MLX `conv3d` followed by a separate FP16 bias add rounds twice. Where a
  group norm and SiLU precede the convolution, the input packing applies them,
  rounding the normalized input to FP16 once as the separate kernel would. In
  residual blocks the projected time embedding and the skip branch are added to
  the same FP32 sum, so each block output is rounded once instead of after each
  addition. Convolutions of a nearest 2x upsampling (the UNet and VQ decoder
  upsamplers) run as eight 2x2x2 parity convolutions of the source, 8/27 of the
  products; their merged FP16 weights are rounded once from FP32 sums. Other
  Macs, and `LESIONDIFFUSION_NAX_CONV=0`, use MLX `conv3d`.
- LMNet's FP32 3x3x3 convolutions run on the same kernel in an exact mode: each
  FP32 input and weight is split into an FP16 high part and a `2^11`-scaled FP16
  low part, and the high-high and the two cross products are accumulated in
  FP32. Only the low-low product is dropped, about `2^-22` relative, so results
  stay at FP32 accuracy (measured below MLX's FP32 `conv3d` error against
  float64) rather than TF32's `2^-11`; outputs stay FP32. Exact mode needs
  values within FP16 range. Layers with at most 16 output channels use
  16-column tiles.
- Pointwise convolutions run as a matmul on the NCDHW layout, without layout
  transposes; like `conv3d`, the product is rounded before the bias add.
- FP16 linear layers run as an FP16 GEMM with FP32 accumulation and bias
  (`addmm`), rounding once like source mode's FP32 promotion without its FP32
  matmul.
- Group normalization uses FP32 chunked moments merged with Chan's update and a
  single vectorized affine pass (`medmlx_core.group_norm`). Where SiLU follows,
  it is applied to the FP32 normalized value before the one rounding.
- LMNet's 32-wide FP32 self-attention heads use an unpadded flash-attention
  kernel (`_attention`). On the M5 family it runs both products on the neural
  accelerators with the same exact FP16 split, including the softmax
  probabilities, and an FP32 online softmax; elsewhere, or with
  `LESIONDIFFUSION_NAX_CONV=0`, an FP32 SIMD kernel. Other attention calls use
  MLX fused scaled dot-product attention with the 32-wide heads zero-padded to
  64 channels, keeping the original 32-channel scale and discarding the padded
  output channels. LINet's FP16 attention therefore uses the fused kernel, whose
  softmax runs in FP32.

medmlx-core disables TF32 (`MLX_ENABLE_TF32=0`) by default, so LMNet's
remaining FP32 matmuls run in full FP32, and its convolutions and attention at
FP32 accuracy as above. Sampling retains the same
timesteps, eta=1, random draws and per-step synchronous evaluation. Source mode
uses uncompiled denoisers, explicit attention, MLX convolution for LMNet's FP32
layers, and the source-ordered FP16 convolution and group normalization.

Use `inference_mode="source"` for the upstream-reproduction path: LINet context
is disabled, its unused text encoder is not evaluated, and FP16 convolution uses
source-ordered arithmetic. For example:

```python
result = run(
    image_path="ct_1mm_ras.nii.gz",
    organ_mask_path="liver_mask.nii.gz",
    output_dir="source-output",
    organ="liver",
    lesion_type="tumor",
    bbox_size_mm=[30, 30, 20],
    placement="inside_parenchyma",
    inference_mode="source",
)
```

The runner returns `outputs` and JSON-safe `details`. Output roles are
`synthetic_image` (`synthetic_ct.nii.gz`, float32 HU), `target_mask`
(`lesion_mask.nii.gz`, uint8 lesion 1) and `organ_mask` (`organ_mask.nii.gz`, uint8
organ 1). The affine, qform and sform of the standardized grid are kept. Details
record schemas, source revision, inference mode, arithmetic, conditioning, stage
timing, selected center and QC. Existing files require `overwrite=True`. Outputs
are written to hidden temporary files and moved into place only after all writes
succeed. A failure before the final moves (for example an LMNet QC rejection)
leaves no new outputs that block a retry. A failure or interruption during the
final moves can leave partial outputs.

The configured v2 route encodes and decodes the full CT and substitutes the
sampled latent crop before decoding. The full reconstructed CT is the output;
voxels outside the lesion can change. With the default `output_region="full"` there
is no extra cosine box blend, mean-density correction or boundary smoothing.
`output_region="box"` instead keeps the generated CT over the box, widened to the
lesion's bounding box when the generated lesion extends past it, so every
lesion-mask voxel lies on generated intensities. It blends that region into the
input CT with weight `0.5 * (1 + cos(pi * d / 6 mm))` (`d` is the distance in mm
from the region; `LOCALIZATION_MARGIN_MM`), retains farther voxels from the standardized input cast
to float32, and writes the full reconstruction as `raw_synthetic_ct.nii.gz`.
The compositing record calls this `original_ct_cast_float32`; higher-precision
input values may round. Full-volume execution can require
substantial memory; it does not use a slab-encoding approximation.

## Upstream decisions and qualification

The upstream repository contains both v1 and v2 inpainting paths. The port uses
the v2 target selected by its inference configuration, with the mode differences
described above. That path passes NumPy reconstructions to `AlignDataset`, whose
NumPy writer applies the inverse CT
transform without the corrections in its separate Torch writer branch.

Preprocessing's hollow-organ set excludes gallbladder, while inference text's
hollow-organ set includes it; each stage follows its own upstream set. With the
supplied organ map used for both `seg` and `coarseg`, upstream LMNet conditions
solid organs as protruding. Source mode retains LINet's `skip_context=True`;
conditioned mode enables the checkpoint's text context path.
Phenotype text conditions the generator; it does not enforce density direction
in the output.

Source-mode seeded, reduced-width Torch CPU reference recordings cover preprocessing,
UNets, VQ encode/quantize/decode, BERT attributes, explicit-noise DDIM,
component QC and v2 end-to-end generation. They do not qualify real
checkpoints, full-size volumes or clinical usefulness. The 104-test suite
passed on M1 Max Metal with MLX 0.32.3 on 2026-10-08, with its existing bounds.
Both an earlier pinned profile (Transformers 5.10.4, NumPy 2.3.5) and the
declared reference extras (Transformers 5.19.0, NumPy 2.5.3) passed.

Discrete geometry, component selection and deterministic VQ quantization
comparisons are bitwise. Cross-framework learned outputs use
provisional float32 `atol=2e-6, rtol=2e-5` and float16 `atol=2e-3, rtol=2e-3`
bounds for reduction order and half rounding; the production-width float32 UNet
test retains `atol=3e-5, rtol=3e-4`. UNet SiLU computes in float32 before rounding
to half, and its softmax accumulates in float32, following Torch. VQ follows its
separate upstream sigmoid/product rounding. Activation tests allow one half ulp
and RMS error of at most 0.1 ulp to cover rounding midpoint differences. Inverse
CT comparisons allow four float32 ulps at 1000 HU (`0.000244140625` HU), with no
relative tolerance near zero.

Source-mode FP16 convolutions retain Torch CPU's channel-major, four-part accumulation,
including its remainder order, and add bias before rounding to half. Source-mode
FP16 GroupNorm uses source-ordered Welford chunks and cascade merges;
complete 128-chunk blocks execute in parallel while retaining the original carry
and residual fold ordering. Its reciprocal root is compensated before the FP32
affine, preserving the source's fused bias
calculation and separate output multiply/add. These kernels execute on Metal;
FP32 layers continue using the shared array primitives. The adapted CPU
algorithms follow PyTorch 2.13's
[`BlasKernel.cpp`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/BlasKernel.cpp),
[`moments_utils.h`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/moments_utils.h) and
[`group_norm_kernel.cpp`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/group_norm_kernel.cpp);
their BSD notice is included in `THIRD_PARTY_NOTICES.md`.

VQ nearest-code selection is discontinuous. This seeded generation fixture has
16 tied distances; a `0.0001` latent perturbation in upstream changes CT by up to
65.19 HU. A uniform 8 HU comparison of independently computed full generations
does not establish parity. The runner reference test fixes upstream encoded,
sampled and quantized stages to check normalization, masks, crops, latent
assignment, native VQ decoding and file writing. Its 8 HU decoder bound maps the
`0.002` normalized absolute bound through the maximum 4000 HU slope. Fully native
generation is tested separately for finite, clamped output and repeatability.
Host checks alone do not qualify native numerical parity.

On 2026-10-08, matched small-volume real-checkpoint comparisons covered
LMNet, LINet, VQ and text.
All four pass against separately recorded Torch 2.13 CPU oracles at the pinned
upstream revision. VQ encoder and continuous decoder outputs match bitwise in
that profile, and code-selection replay has no changed sites. The VQ oracle
records raw selected codebook vectors; the source model retains its FP16
straight-through arithmetic, whose rounding is covered by the recorded bounds.
The Torch 2.8 VQ fixture failed at one code selection. These small-volume
comparisons and the seeded runner do not establish full-volume Torch/CUDA
generation parity or clinical usefulness.

Before the tiling change, the synchronized warm convolution benchmark on
2026-10-08 measured 44.4 ms for the source-ordered kernel versus 1.59 ms for the
shared FP16 convolution on this M1 Max, input `[1,96,16,24,24]` and kernel
`[96,96,3,3,3]` (two warmups, three samples). That historical operator was about
28 times slower; it did not predict whole-model latency.

On 2026-10-09, shared input and weight tiles preserved the four source partial
sums and final rounding. All 12 paired operators and four paired real-checkpoint
network forwards remained bitwise
identical. Larger operators improved 10.7–14.9 times; LINet at 32³ improved from
9.133 to 0.717 s. VQ encoding of a 64³ image and decoding a 32³ latent
into a 64³ image improved 13.0/12.8 times. Seven tiny operators were slower
from tile overhead. The recorded source-mode LINet 64³ median is 5.655 s;
its older 85.24 s diagnostic was not repeated, so that comparison is unpaired.
The package suite passed 113 tests without skips or changed tolerances.

The historical source-mode 200-step run on the matched 160×160×128, 1 mm RAS AMOS
case 0008 crop completed in 22.95 minutes with 18.74 GB peak MLX memory. Output geometry was
preserved, CT was finite in [-1000,1000] HU, and all 6,191 lesion voxels were
inside the liver mask. The requested hypodensity failed its informational
contrast check: lesion mean 73.4 HU versus 66.8 HU in surrounding liver. Full VQ
reconstruction changed voxels outside the lesion by a mean absolute 15.4 HU
(99th percentile 145.1 HU). This runtime/QC result does not qualify phenotype
control or full source generation parity, and does not measure the conditioned
mode. The original 292×230×204 prepared volume remains unqualified; the attempt on
it was interrupted in favor of the matched crop.

## Regenerating reference fixtures

Recordings are loaded directly from `tests/fixtures/darwin-arm64/`. The package
supports Apple Silicon Macs only. Torch reference computations run on the Mac's
CPU; native package execution requires Metal. Every NPZ carries `schema_version`,
full upstream and text revisions, `recording_platform="darwin-arm64"`, reference
framework and version metadata, and is well below 1 MB. No real checkpoint is needed.

On another Apple Silicon Mac, install the extras above, fetch the pinned
LesionDiffusion and OpenCLIP files outside the checkout, then run both recorders:

```sh
"$TMPDIR/venv-lesiondiffusion-mlx/bin/python" scripts/fetch_reference.py \
  --output-dir "$TMPDIR/lesion-upstream"
PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/venv-lesiondiffusion-mlx/bin/python" \
  scripts/make_reference_fixtures.py --upstream-root "$TMPDIR/lesion-upstream"
PYTHONDONTWRITEBYTECODE=1 "$TMPDIR/venv-lesiondiffusion-mlx/bin/python" \
  scripts/make_runner_fixtures.py --upstream-root "$TMPDIR/lesion-upstream"
"$TMPDIR/venv-lesiondiffusion-mlx/bin/python" -m pytest -q
"$TMPDIR/venv-lesiondiffusion-mlx/bin/ruff" check .
```

The recorders execute upstream numerical definitions.
An AST loader omits optional training, plotting, service and segmentation imports;
training base classes are replaced with Torch modules, CUDA-only buffer storage
is explicitly redirected to CPU, and seeded tiny models replace pretrained
constructors. Numerical method bodies are unchanged. Tiny fixtures retain their
input/weight seeds, with architecture shapes corrected to the source contracts.
BERT uses Transformers eager attention and upstream OpenCLIP's CLS pooler and
bias-free MLP projection. The `reference` extra pins the available recording
frameworks; it is separate from upstream's original training environment.
Fixture tests load recordings and never import the recording scripts. Additional
Metal source-block tests use the fetched external source directory at
`$TMPDIR/lesion-upstream`, or an explicit `LESIONDIFFUSION_UPSTREAM_ROOT` directory.
A missing recording is a failure, while unavailable Metal is reported as a
native-test skip. The recorders reject hosts outside macOS arm64.


`fetch_reference.py` downloads the files from the revisions declared in
`reference.json`; its `--verify-only` option checks file presence. The recorders
and live source tests execute the supplied source directory directly. Keep that
directory at the declared revisions when regenerating reference fixtures.

## Static checks

`make qa` runs Ruff and standard Pyright with the existing project interpreter and
maintained host Node. It does not run inference or synchronize the environment.
`make env` explicitly provisions the locked development environment when needed;
an existing external environment can be used with `make qa PYTHON=/path/to/bin/python`.
Run Metal and source-oracle tests locally as an explicit validation step, with
the fetched source directory configured above.
