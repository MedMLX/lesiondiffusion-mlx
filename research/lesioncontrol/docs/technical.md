# Reference control and numerical contracts

LesionControl uses the parent LesionDiffusion MLX generator's supplied-mask LINet
path, DPM++ 2M sampler and reference FP32 decoder. The model receives only an
independently clean host. Reference pixels enter feature extraction and training
cache preparation; they are not supplied as generation images.

The controller varies retained schedule steps, density, homogeneity, margin,
numeric-conditioning schema, numeric offset and HCC description. A bounded
SciPy differential-evolution search fits a reference-specific policy. Short
schedules use image-to-image initialization with the clean latent and noise at
the selected starting time. Solver history starts empty. Full-strength sampling
retains the production solver's trajectory.

## Appearance measurement

Profiles contain 101 appearance features: 21 first order, 26 GLCM, 16 GLRLM,
16 GLSZM, 17 GLDM and five NGTDM features. Extraction uses 25 HU bins, a -1000 HU
bin anchor, 1 mm spacing, and no additional normalization, resampling, filtering
or resegmentation. Canberra errors are averaged within each family and then
equally across families. The texture endpoint uses the five non-first-order
families. Undefined features reject the input or trial.

Shape features are excluded from appearance fitting. The supplied outline is
fixed, and some appearance features still depend on ROI size. Matching its
features evaluates conditional appearance fidelity.

`absolute_hu` profiles retain unshifted HU. `liver_contrast` profiles subtract the
reference's 3–10 mm non-tumor liver-ring mean before extraction and discretization.
Generated features subtract the unchanged clean-host ring mean. Numeric text
conditioning translates the requested contrast into the host's absolute HU
space. Changes to generated liver attenuation cannot move this target.
Acquisition timing is an external input; a single-volume contrast measurement
does not determine arterial/portal enhancement dynamics.

The reliability runner additionally records actual generated-ring contrast and
the change in ring attenuation. A second feature extraction uniformly shifts an
in-memory liver-centered image until its lesion mean matches the reference.
This mean-aligned diagnostic probes attenuation and bin-origin sensitivity;
remaining differences still involve histograms, discretization and spatial
structure. It leaves the saved synthetic CT unchanged.

## Learned conditioning

The adapter maps 101 training-normalized features through a 64-unit SiLU MLP to
distinct residuals for ten existing 512-dimensional context tokens. It contains
339,328 trainable parameters and starts with a zero final projection. Feature
means and scales use training patients only, with equal total weight per patient.
Constant scales are one and feature values are not clipped.

Residuals are applied before LINet attention K/V caching. The backend scopes and
restores its process-local hooks, serializing its own generation calls with a
reentrant lock. Gain zero preserves the same ten-field text schema, requested
numeric HU and sampler controls while disabling only the learned residual.
Omitting an adapter instead uses the native text schema.

The MLX preflight checks exact predictions at zero initialization, finite nonzero
adapter gradients, and a directional finite difference through a frozen LINet.
The CUDA bridge compares normalized features, context residuals and frozen-model
predictions against an exported fixture. Production gain-zero comparison checks
all output images, masks and affines. These checks have distinct roles: adapter
arithmetic, model integration and public generation behavior.

## Training and source preparation

CT is cast to float32 before linear interpolation; masks use nearest-neighbor
interpolation. Native component identity, source acquisition and grid checks
precede prepared 128³, 1 mm RAS crops. The preparer verifies supplied provenance
assertions and does not recreate raw DICOM conversion or acquisition proofs.
The [research guide](research.md) defines the required external inputs.

MLX exports frozen real-image VQ latents and text embeddings. The separate CUDA
trainer uses full 64³ latents, FP32, patient-balanced sampling, AdamW and gradient
norm clipping at one. The source loss sums masked epsilon-L1 over four latent
channels, divides by the latent lesion bounding-box voxel count plus epsilon,
and adds auxiliary mask Dice loss. Only the four MLP parameter tensors are
optimized. Checkpoints preserve optimizer and RNG state for continuation.

Generation hosts must have no annotated Mass anywhere in the full model input,
including outside the placement region. Conditional host preparation requires
acquired source coverage, the rigid reference outline and a 3 mm margin inside
provided liver, and at least 1,000 non-tumor ring voxels. It never fills existing
tumors or resizes outlines. Geometry-only bounded searches retain exclusions.

## Paired evaluation

Checkpoint, gain, text, numeric target, schedule and host placements are fixed
before evaluation. Each gain-zero/gain-one pair uses the same seed and supplied
outline. Reference features remain declared conditioning inputs. The frozen
study runner uses 20 paired seeds per reference and preserves failures and
attempt events without substituting seeds.

Evaluation retains every feature error, six family means, contrast direction,
numerical sign, target crossing and simultaneous endpoint improvements. Patient
means are cohort analysis units; seeds are technical repetitions. Patient-cluster
bootstrap intervals condition on the fixed seed panel and registered host
mapping. Shared donor anatomy limits host-population inference.

Selection references, adapter-development holdouts and previously evaluated
references remain separate. A holdout from adapter development does not establish
absence from base-generator pretraining. Alias and constant feature rows are
flagged, and correlated feature counts are descriptive without feature-wise
significance tests.

Run records, patient manifests, images, weights and generated feature tables are
external to the source distribution. The analysis entrypoint reads completed
outputs and requires all registered pairs unless partial reporting is requested.
