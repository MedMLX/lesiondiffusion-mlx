# Inference performance and numerical contracts

The conditioned pipeline runs LMNet in FP32, LINet and VQ encoding in FP16,
and VQ decoding in FP32. The decoder retains FP32 because its upsampling
activation can exceed FP16's finite range. Source mode keeps its original
arithmetic. The default sampler remains the upstream uniform, stochastic
200-step DDIM schedule with eta 1.

## Fast inference

`sampler="dpmpp_2m"` selects deterministic DPM-Solver++ 2M. It uses exactly
the requested number of evaluations for each stage. `lmnet_diffusion_steps`
and `linet_diffusion_steps` can override the shared `diffusion_steps` value.
The 200-step DDIM path remains the reference when these options are omitted.

```python
from lesiondiffusion_mlx.medmlx import run

result = run(
    image_path="ct.nii.gz",
    organ_mask_path="liver.nii.gz",
    weights_path="weights",
    output_dir="generated",
    organ="liver",
    lesion_type="tumor",
    bbox_size_mm=(30, 30, 20),
    placement="inside_parenchyma",
    sampler="dpmpp_2m",
    diffusion_steps=10,
    decoder_mode="scaled_fp16",
    output_region="lesion",
)
```

The optional scaled decoder requires enabled NAX. It retains FP32 codebook
selection, then uses half weights and activations. Upsampling and subsequent
residual branches scale before half storage; normalization of scaled inputs
uses the correspondingly scaled epsilon. This prevents the known 75,000-range
overflow without clipping activations. The reference decoder remains available.
LINet loads after VQ encoding finishes, and decoder weights load after LINet
is released. These stage lifetimes reduce the peak allocation without changing
the sampled latent or codebook selections.

Measured complete consumer calls on the reference workload include model
loading, compilation, sampling, both decodes and NIfTI writing. Warm medians
use five timed calls after warmup for the reference configurations and three
for the final scaled configurations. Memory is peak MLX allocation in decimal GB.
This measures active arrays, separately from reusable allocator cache and total
process memory.

| Workflow | Cold | Warm median | Peak active MLX |
| --- | ---: | ---: | ---: |
| Generated mask, DDIM200, original capture run | 55.01 s | — | 5.19 GB |
| Generated mask, DPM25, reference decoder | 8.70 s | — | 5.19 GB |
| Generated mask, DPM10, reference decoder | 4.57 s | 4.70 s | 5.19 GB |
| Generated mask, DPM10, scaled FP16 and cache cap | 3.63 s | 3.52 s | 2.60 GB |
| Supplied mask, DDIM200, reference decoder | 41.66 s | — | 5.19 GB |
| Supplied mask, DPM5, reference decoder | 2.98 s | 2.97 s | 5.19 GB |
| Supplied mask, DPM5, scaled FP16 and cache cap | 2.18 s | 2.01 s | 2.60 GB |

The original 55.01-second run includes component capture. The comparison below
uses capture-free DDIM measurements to isolate the complete inference call.
Final scaled-decoder warm ranges were 3.50–3.53 seconds for generated masks
and 2.00–2.03 seconds for supplied masks. Deferring model loads reduced the
scaled configuration's peak from 3.30 to 2.60 GB; latency was similar for
generated masks and varied more for supplied masks.

A later comparison separated precision and loading policy in one process.
Each configuration had one warmup followed by three timed complete calls,
with configuration order rotated between rounds. No intermediate-array or
process-footprint sampling ran inside these timed calls.

| Decoder | Loading policy | Warm median | Peak MLX allocation |
| --- | --- | ---: | ---: |
| Reference FP32 | Eager | 4.896 s | 5.186 GB |
| Reference FP32 | Deferred | 4.897 s | 5.186 GB |
| Scaled FP16 | Eager | 3.983 s | 3.301 GB |
| Scaled FP16 | Deferred | 4.035 s | 2.604 GB |

The precision change accounts for the observed latency improvement. Deferral
reduces the scaled path's memory peak without a consistent latency benefit.
Before cache tuning, the scaled configuration's warm medians ranged from
3.66 to 4.08 seconds between measurement series; the interleaved comparison
measured 4.04 seconds.

Separate synchronized allocator traces showed active MLX allocation falling
from 746.812 MB before sampling to 36.170 MB after deleting LINet and its
sampler. Nominal retained weight-array storage fell from 735.449 to 28.968 MB.
Clearing the allocator cache then left active allocation unchanged; decoder
preparation raised it to 51.489 MB. Pipeline return left 16 bytes of active MLX
allocation and 3.048 GB of cache; an explicit cache clear emptied the cache.
These counters describe array lifetimes and reusable MLX cache, separately
from process footprint or operating-system memory release.

Before cache tuning, interval-sampled process physical footprint was
11.337 GB for eager FP32 and 11.168 GB for deferred scaled FP16, about 1.5%
lower. Released sampling buffers accumulated in MLX's free cache, which
reached 10.681 GB in the instrumented scaled run before its stage-boundary
clear. The 49.8% reduction in peak active MLX allocation therefore did not
mean that total RAM was halved.

The native MLX free-cache limit was evaluated in fresh processes at the
default limit and at 4, 2, 1 and 0 GiB. Every saved array and affine was exact
across all 23 screening and confirmation calls. Four GiB gave the largest
process-memory reduction without an established latency cost: confirmation
physical-footprint peaks were 11.124 GB at the native default and 6.907 GB
with the cap, a 37.9% reduction. Active peak allocation remained 2.604 GB.
Confirmation warm medians were 3.606 and 3.491 seconds respectively; screening
reversed the small timing difference. Lower cache limits reduced memory
further but increased the measured latency.

The native API is
[`mlx.core.set_cache_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html).
It limits reusable free buffers rather than live arrays. Scaled-decoder runs
scope a 4 GiB cap to native generation, preserve an existing tighter caller
limit and restore the prior limit on return or failure. The measured process
footprint is an interval-sampled physical-memory metric, distinct from active
array allocation.

Public-run integration measurements used no external cache setter. Generated
DPM10 measured 3.627 seconds cold, 3.522 seconds warm median and a 6.907 GB
process-footprint peak. Supplied-mask DPM5 measured 2.176 seconds cold,
2.008 seconds warm median and a 6.896 GB process-footprint peak. Both retained
the 2.604 GB active peak, recorded the 4 GiB applied cap and reproduced every
saved array and affine exactly across eight calls. Public runner tests cover
preserving tighter caller limits and restoring them after native failures.

Changing sampler or step count changes the generated trajectory. On the
initial case, DPM10 versus DPM25 had mask Dice 0.9899 and centroid displacement
0.086 mm. Those describe convergence between DPM configurations, rather than
agreement with stochastic DDIM or image realism. The scaled decoder retained
the exact mask and codebook selections; lesion HU mean absolute error versus
the FP32 decoder was about 0.12 HU, with a maximum below 0.5 HU. Whole-volume
maximum error was 2.18 HU; distant input voxels were still exact.
The maximum was outside the lesion, at voxel (32, 47, 34), exactly 33 voxels
from the mask in per-axis distance and 50.78 mm in Euclidean distance. Within
the permitted region, absolute-error P95/P99/P99.9 were 0.306/0.345/0.378 HU;
within the lesion they were 0.270/0.333/0.402 HU. Both decoder outputs were
bitwise unchanged beyond the permitted region.

The scaled decoder was then compared with the reference decoder across all
ten DPM10 conditions in the two-subject study below. Masks and affines matched
exactly, every output was finite, and both configurations preserved all input
voxels beyond the permitted region. Lesion MAE ranged from 0.1137 to
0.1163 HU; lesion maximum error ranged from 0.3934 to 0.4786 HU.

| Absolute-error metric | Lesion: largest per-condition value | Permitted region: largest per-condition value |
| --- | ---: | ---: |
| P95 | 0.277 HU | 0.316 HU |
| P99 | 0.345 HU | 0.382 HU |
| P99.9 | 0.441 HU | 2.200 HU |
| Maximum | 0.479 HU | 3.852 HU |

The largest observed error was outside the lesion in the second subject, at
voxel (120, 79, 95): input 544.632 HU, reference 544.686 HU and scaled
548.538 HU. Its distance from the lesion was 32 voxels per axis at most
(32.98 mm Euclidean), inside the permitted region. The HU transform maps
normalized decoder values above 0.8 with slope 4000 HU per unit, six times
the slope used for -200 to 200 HU. High-attenuation voxels therefore amplify
normalized decoder errors more strongly than values in the central HU range.

Intermediate stability captures covered generated and reconstruction decodes
for the initial condition and the condition with the largest HU error.
All 88 recorded stored tensors were finite. Scaled upsampling maxima were
11,216 and 11,736; the largest recorded activation was 11,808, below FP16's
finite limit of 65,504. Four direct FP32 selected-latent-to-half casts covered
6,553,600 elements with no nonzero values becoming zero. Stored half
subnormals, down to 2^-24, and some zeros were observed. These captures
characterize stored representations; fused arithmetic temporaries were not
separately observed. Existing native tests also exercise a 75,000-valued
FP32 branch scaled before half storage and residual addition.

The solver reference is Diffusers 0.42.1 at revision
`2a1433913c7044f34fa4767dd84b28bba0bb0ecf`: epsilon prediction, second-order
midpoint updates, rounded linspace timesteps, no thresholding and terminal
sigma zero. Solver state and history are FP32, including promoted LINet noise
predictions. The terminal update uses the finite first-order x0 limit. Compact
independent Torch CPU recordings cover both networks' stored alpha curves,
masked and unmasked trajectories, and 4, 8 and 25 steps. The fixture recorder
is `benchmarks/record_dpm_reference.py`; Diffusers is not a runtime dependency.

## Repeated inference

Conditioned inference prepares a compiled DDIM step that includes input
construction, the denoiser and the update. Random draws remain outside the
compiled graph, with one initial draw and one draw per step, including when
eta is zero. Each step evaluates before the next to bound activation memory.
The source sampler remains available for source reproduction.

Each denoiser binding evaluates the fixed text projection and cross-attention
keys and values once. A binding owns these arrays and assumes its weights and
dtype remain fixed. Self-attention always uses the evolving image. Bindings
are recreated for each request; the public runner still loads models and
constructs compiled callables for each invocation.

On supported M5 hardware, FP16 self-attention uses 32-wide NAX heads directly.
Q/K products, online softmax state and output accumulation use FP32;
probabilities supplied to NAX and the returned output round to FP16.
Short-context attention retains padded MLX SDPA. The original FP32 attention
path remains in place. `LESIONDIFFUSION_NAX_CONV=0` disables both NAX
convolution and NAX attention.

Four-channel native VQ search keeps each position's best distance and first
matching index in a SIMD group. It avoids the position-by-codebook distance
matrix, preserves dtype rounding and the source's `z + (selected - z)`
expression, and reuses codebook norms until the codebook array is replaced.
Other configurations retain chunked lookup. Codebook arrays are immutable
during an inference binding.

## Sampler, morphology and spatial changes

A comparison used two AMOS CT subjects, with five matched seed, box-size and
location conditions per subject. Each condition ran DDIM200, DPM25 and DPM10
with the reference FP32 decoder and component capture disabled: 30 complete
runs. The ten conditions are repeated observations within two subjects.
The scaled-decoder follow-up used the same ten DPM10 conditions, reusing the
matching initial scaled result and running the other nine conditions.

| Method | Median cold time (range) | Peak MLX allocation |
| --- | ---: | ---: |
| DDIM200 | 52.19 s (49.43–56.19) | 5.19 GB |
| DPM25 | 8.38 s (7.67–8.71) | 5.19 GB |
| DPM10 | 4.77 s (4.51–5.10) | 5.19 GB |
| DPM10, scaled FP16 and deferred loading | 3.83 s (3.67–4.04) | 2.60 GB |

The median matched DDIM200/DPM10 time ratio was 11.01 with the reference
decoder and 13.78 with the scaled decoder and deferred loading. These compare
complete cold calls; the follow-up measurements were collected separately
from the initial sampler study. DPM25 and DPM10 masks
had median Dice 0.9864 (range 0.9363–0.9938), median centroid separation
0.066 mm (maximum 0.831 mm), and pooled bidirectional boundary HD95 at most
2 mm. CT RMSE within the union of their lesion masks had median 5.76 HU.
These quantify agreement between the two DPM schedules.

LMNet generates a lesion mask before LINet synthesizes CT conditioned on that
mask. For thresholds 0.9 through 0.3, component selection keeps the largest
face-connected component and chooses the threshold whose retained volume is
closest to the requested box volume. Thus sampler and component selection
both affect the mask passed to LINet; generated CT attenuation cannot cause
an earlier LMNet mask to change.

Retained morphology differed between DDIM and DPM. For one condition,
DDIM200/DPM25/DPM10 voxel volumes were 4.443/2.693/2.786 mL and sphericities
were 0.694/0.547/0.565. All saved masks were one face-connected component,
as imposed by component selection. Mesh facets display voxel geometry.
Surface-to-volume ratio also depends on lesion size. AMOS organ labels do
not supply reference lesion annotations for these synthetic insertions.

Replaying all 30 LMNet stages with raw capture reproduced every final mask
exactly. At the source-selected threshold, median raw component counts were
1/81.5/115 for DDIM200/DPM25/DPM10; median retained fractions of thresholded
foreground were 100%/96.20%/94.37%. Every case had exactly one component of
at least 50 mm³. The other DPM components were small islands, whose combined
discarded volume reached 1.481/1.730 mL. In the seed-25 condition, DPM25 and
DPM10 retained 71.69% and 70.19% of foreground.

All DPM conditions selected threshold 0.3; DDIM selected thresholds 0.3–0.7.
The continuous LMNet scores are not calibrated probabilities. Comparing a
fixed cutoff of 0.5 reduced DPM25/DPM10 median component counts to 3/4, with
median retained fractions above 99.9%. DPM25/DPM10 raw-mask Dice remained
high before component selection: median 0.9788 at the selected cutoff and
0.9859 at cutoff 0.5. At cutoff 0.5, raw DDIM/DPM10 Dice was only 0.5775.
Thus the sampler-dependent geometry persists before component selection and
under a common cutoff; the reported island counts also depend strongly on
threshold. No threshold or component-selection behavior was changed.

Attenuation was evaluated both within each generated lesion and in a fixed
ROI shared by the methods: requested box intersected with the input liver.
Median mean-HU changes in that fixed ROI were -11.00/-9.06/-10.06 HU for
DDIM200/DPM25/DPM10. Absolute lesion-to-background CNR medians were
2.76/3.03/3.14, using a common liver background 3–10 mm outside the union of
the generated masks. Per-condition contrast changes varied in direction.
Neither CNR nor agreement with another sampler establishes lesion realism.

All 30 volumes were finite and bitwise unchanged beyond the mask's defined
33-voxel per-axis neighborhood. Inside the 1–5 mm Euclidean boundary shell,
median mean absolute CT changes were 3.04/3.93/3.90 HU. Maximum changes
outside the lesion mask reached 56.08/65.07/59.65 HU across conditions, so
the low whole-volume mean does not describe every boundary voxel. The
existing spatial blending is retained.

A prompt-free nnU-Net liver/cancer model, Dataset003_Liver fold 0 with
eight-flip augmentation, was applied to the original CT and all three
seed-26 outputs from the first subject. It received no generated mask or
box prompt. Within the fixed requested box intersected with the input liver,
cancer-label Dice against each intended mask was 0.827/0.707/0.757 for
DDIM200/DPM25/DPM10; predicted volumes were 5.343/1.780/2.904 mL. The original
CT produced no cancer-labelled voxels. This measures independent model
agreement on one partial CT crop; manual lesion annotations were unavailable.

## Measurement protocol

The reference workload is AMOS case 0008, a RAS 160 × 160 × 128 volume at
1 mm spacing, liver tumor, seed 23, conditioned inference and 200 DDIM steps.
Generated-mask runs use a 30 × 30 × 20 mm box centered at voxel (80, 80, 64).
Lesion output includes both decoder passes and NIfTI writing.

The paired qualification uses an Apple M5 Max with 128 GiB unified memory, macOS 27.2,
Python 3.13.17, MLX 0.32.4 and medmlx-core 0.2.6. Comparisons serialize GPU
work, evaluate completed outputs, and alternate baseline and candidate in
fresh processes. Cold calls include graph compilation; subsequent calls warm
runtime caches while still recreating the models, as the public API does.
Isolated denoiser measurements separate first-call compilation from warmed
execution. Peak MLX allocation and process footprint are different metrics.
Earlier Metal captures, tile and decoder experiments, and precision and
schedule exploration used medmlx-core 0.2.5. Its relevant runtime, convolution
and normalization sources match the 0.2.6 qualification runtime.

The benchmark helpers live in `benchmarks/`. `pipeline.py` accepts real NIfTI
inputs, optional supplied lesion masks, output regions and component capture.
`pipeline_case.py` exposes the public pipeline to a function benchmark runner.
Its evidence sidecars record the source revision and complete development
patch, runtime configuration, stage records and actual step counts. Capturing
component arrays adds synchronization and I/O, so capture runs are separate
from throughput measurements.
With `--capture-components`, each `lmnet-qc-attempt-N.safetensors` stores
continuous scores before thresholding, the reference box and the selected
component. `linet-lesion-mask.safetensors` stores the full padded mask supplied
to LINet in source and RAS array order. These host-boundary captures work for
both samplers and include every LMNet QC attempt.

`attention.py`, `vq.py`, `components.py` and `sampling.py` provide component
comparisons. The sampling cadence sweep is exploratory and uses a fixed-noise
trajectory; it does not replace measurements of the production sampler.

## Correctness

The 200-step generated-mask workload produced identical saved synthetic CT,
raw reconstruction, lesion and organ masks, spatial affines and final MLX RNG
state against the retained original conditioned implementation. This checks
one case and seed. Compiled arithmetic and native FP16 attention can change
intermediate floating-point values, so endpoint equality on this case does
not establish bitwise equality for every trajectory.

Attention tests compare against independent FP64 softmax calculations. FP32
retains its 2e-5 absolute bound. FP16 uses the existing model-equivalence
bound, absolute and relative tolerance 2e-3. VQ selection and straight-through
values require exact agreement, including ties and strided batches.

Lesion compositing still keeps the original CT outside its defined
33-voxel neighborhood. Full and box output contracts are unchanged. Finite
checks remain at sampled-latent and output boundaries.

## Hardware findings and rejected candidates

A sustained LINet Metal System Trace measured 99.079% selected-process GPU
activity within its active span, with no idle gap of 1 ms or more. The CPU was
queuing work ahead of the GPU. This supports targeting work performed on the
GPU rather than assuming substantial host starvation.

The captured forward contained 485 dispatches, including native NAX
convolution and MLX's NAX SDPA. Head padding and skip concatenations generated
real copy dispatches. Their logical byte counts describe copies, rather than
measured DRAM traffic. Shader timings and hardware bandwidth/arithmetic
counters were unavailable on this profiling configuration, so compute versus
bandwidth limitation and the physical throughput ceiling remain unresolved.

Several measured candidates were retained only as experiments:

| Candidate | Finding | Decision |
| --- | --- | --- |
| Batch-two FP32 decoding | About 1.155 versus 1.154 seconds; peak 5.18 versus 10.25 GB | Keep sequential decoding |
| Smaller NAX row tiles | Exact outputs, about 4% slower on the measured LINet forward | Keep existing tiles |
| Two-input skip packing and projection | Exact outputs, lower footprint, no supported throughput gain | Keep existing concatenation |
| Longer evaluation cadence / asynchronous sampling | No established production-path gain; some runs slower | Keep every-step evaluation |
| Selective safetensors loading | MLX already reads only evaluated tensor data; loading and conditioning were a small fraction of runtime | Keep native loading |

MLX 0.32.4's safetensors loader reads the header and constructs lazy tensor
loads. Filtering stages before evaluation avoids reading unrelated tensor
storage. Reopening the bundle repeats header parsing and lazy-node creation,
not a complete read of all model weights.

## Precision and schedule experiments

One-case LMNet experiments kept the 200-step schedule and FP32 sampler state.
FP16 changed 14 mask voxels (Dice 0.998612); BF16 changed 92 (Dice 0.990868).
Neither replaces the FP32 reference. These dtype experiments compare complete
network arithmetic paths and do not isolate weight precision alone.

The reduced-step experiment measured the following morphology changes against
the 200-step generated lesion:

| Requested steps | Actual steps | Mask Dice | Volume change | Centroid displacement |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 25 | 0.5871 | +40.88% | 5.557 mm |
| 50 | 50 | 0.5435 | +8.88% | 5.068 mm |
| 75 | 77 | 0.5887 | +44.73% | 5.559 mm |
| 100 | 100 | 0.5696 | +15.36% | 5.030 mm |
| 150 | 167 | 0.8678 | +8.90% | 1.040 mm |

The upstream schedule can produce more steps than requested. Equal seeds do
not align random trajectories across schedules. These results demonstrate
changed lesions and do not establish quality equivalence. Every measured
output remained finite and retained distant anatomy exactly.

## Remaining work

1. Obtain supported shader timing and hardware counters to distinguish
   arithmetic throughput, memory traffic and occupancy limits.
2. Extend the two-subject sampler study and qualify the scaled decoder across
   additional input geometries, lesion sizes and locations. Compare generated
   morphology with independently annotated real lesions.
3. Evaluate crop-scoped VQ as a separate inference configuration. Spatial
   GroupNorm changes with the crop, and a crop-only path cannot silently
   replace the full raw reconstruction artifact.
4. Consider persistent runtimes for repeated requests when their measured
   loading and compilation cost justifies the retained memory.
5. Evaluate distillation as model-quality research with multiple cases and
   seeds, rather than an equivalent scheduler rewrite.
