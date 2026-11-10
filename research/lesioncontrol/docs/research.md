# Research entrypoints

The research workflow prepares provenance-checked NIfTI crops, exports frozen
MLX latents and text contexts, trains a small context adapter on CUDA, and
evaluates decoded lesions on MLX Metal. Run the commands below from this
LesionControl project directory, using the environment described in the README.

Native CT, binary Mass and liver segmentations, source catalogs, outline metadata
and frozen patient partitions are external inputs. Raw TCIA/DICOM conversion and
a maintained adapter-split builder are outside this directory. The preparation
scripts consume the contracts below; they do not construct those upstream inputs.
Keep medical images, derived caches, weights, patient manifests and generated
results outside Git.

| Entrypoint | Purpose |
| --- | --- |
| `python -m lesioncontrol.training_data` | Check native acquisition/segmentation provenance and prepare real training/selection crops. |
| `python -m lesioncontrol.training_cache` | Export real-image VQ latents, ten text tokens, radiomics and training-only normalization on Metal. |
| `python research/train_cuda_adapter.py` | Train or resume the adapter against a frozen source LINet on CUDA. |
| `python -m lesioncontrol.adapter` | Run the MLX gradient preflight. |
| `python -m lesioncontrol.learned` | Compare an exported CUDA fixture with the MLX adapter/model path. |
| `python research/evaluate_adapter.py` | Register and run the fixed development checkpoint/gain/step selection protocol. |
| `python research/prepare_heldout.py` | Prepare registered unseen references and search the fixed native host pool. |
| `python research/qualify_heldout_geometry.py` | Refine acquired-FOV crop geometry and search bounded crop banks. |
| `python research/prepare_donor_extension.py` | Apply the registered donor-only extension to previously unplaced outlines. |
| `python research/run_reliability.py` | Register and run paired frozen-adapter draws with original and mean-aligned radiomics. |
| `python research/analyze_reliability.py` | Analyze existing paired outputs on CPU without loading models. |

Use each entrypoint's `--help` for its required arguments. Invoke research scripts
directly as shown: their sibling imports assume this script layout. Run tests
from this project directory with `make test TESTS=tests/test_reliability_analysis.py`.

## Native inputs and frozen partitions

Under `--source-data`, each patient directory contains `source_contrast.nii.gz`,
`tumor_mask.nii.gz` and `liver_mask.nii.gz`. Masks are binary 0/1 on the CT's native
grid. The tumor mask identifies the cataloged Mass segment. Native component
ranking uses 26-connectivity; outline ranks and identities are checked before
resampling. CT uses float32 linear interpolation, masks use nearest-neighbor
interpolation, and prepared crops use a 128³, 1 mm RAS grid. A real reference must
retain one face-connected lesion and at least 1,000 non-tumor liver voxels in its
3–10 mm ring.

The `training_data --split` JSON requires:

| Field | Contract |
| --- | --- |
| `train_patients`, `selection_patients`, `never_train_reserved_patients` | Disjoint arrays of patient identifiers. Preparation admits only training and selection patients. |
| `source_case_catalog` | Path to a JSON object with `cases`. Each case has `id` equal to its patient identifier, `tumor_voxels`, `source_contrast: {series_uid, image_count}`, and `segmentation` source metadata. |
| `source_split` | Path to the original eligibility JSON with `patients`, each containing `eligible_lesion_ids`. |
| `shape_metadata_source` | Path to the outline CSV. Rows require `patient`, `lesion_id`, one-based descending component `rank`, `voxel_count`, and `centroid_r_mm`, `centroid_a_mm`, `centroid_s_mm`. Held-out search also reads `max_diameter_mm`. |

Paths in these input files are interpreted from the process working directory;
absolute paths also work. Component voxel counts must match the catalog exactly,
and physical centroids must agree within 0.011 mm.

Normally the native CT's z length equals catalog `source_contrast.image_count`.
Otherwise an externally established `--proof` JSON must identify the same `case`
and `source_series_uid`, give `selected_ct_shape`, and set
`cached_ct_array_equal`, `cached_ct_affine_equal`, and at least one of
`cached_tumor_array_equal` or `cached_mass_array_equal` to `true`. If both Mass
assertions are supplied, they must agree. The preparer checks these assertions;
it does not reconstruct DICOM or independently create that proof.

```bash
.venv/bin/python -m lesioncontrol.training_data \
  --split inputs/adapter_split.json --source-data inputs/native \
  --proof inputs/acquisition_proof.json --output results/prepared

.venv/bin/python -m lesioncontrol.training_cache \
  --manifest results/prepared/manifest.json --weights "$MLX_WEIGHTS" \
  --output results/training-bundle
```

The prepared manifest has schema `lesioncontrol.real_hcc_training_data.v1` and
`cases` with `id`, `patient`, `partition` (`train` or `selection`), `real_ct`,
`real_mask` and `real_liver_mask`; preparation also records all-tumor masks,
checks and source provenance. Feature normalization uses training patients only,
with equal total weight per patient. A cache produced with `--maximum-cases`
sets `dataset_complete=false` and is accepted only for the CUDA preflight.

## Portable training bundle and CUDA setup

The cache writes `index.json` with schema
`lesioncontrol.adapter_training_bundle.v1`. It contains `dataset_complete`,
`target_space`, 101 ordered `feature_names`, `feature_center`, `feature_scale`,
ten ordered `context_fields`, `schedule_file`, and `cases`. Each case has
`patient_id`, `lesion_id`, `split` and a bundle-relative `file`. Training and
selection patient memberships must be disjoint. The schedule is a finite,
strictly decreasing, positive 1,000-element alpha array bounded by one.

Each case NPZ contains finite float32 arrays: `latent` with shape
`(1, 4, 64, 64, 64)`, `fine_mask` and `coarse_mask` with shape
`(1, 1, 64, 64, 64)`, `context` with shape `(1, 10, 512)`, and `features` with
shape `(101,)`. Masks must be nonempty and binary, with the fine mask inside
the coarse bounding box. Transfer the complete bundle, retaining relative paths,
to the CUDA host. VQ encoding and text encoding remain frozen Mac preprocessing.

The standalone trainer uses its own Python 3.13 environment. The three historical
package versions retained in training records are Torch `2.13.0+cu130`, NumPy
`2.5.4` and safetensors `0.8.0`. The requirements file also includes `einops` and
Pillow because the source LINet import chain requires them; those constraints
follow the parent project's reference/development dependencies. Their historical
installed versions were not recorded.

For the CUDA setup below, use the external LesionDiffusion `ldm` source at
`880736e2b8a4d9becdb6594da7f8af6db19d04d1`, matching the parent package's
`src/lesiondiffusion_mlx/reference.json`. Configure that checkout explicitly;
the PyPI package named `ldm` does not provide this contract.

```bash
export LESIONCONTROL_CUDA_ENV="$HOME/.venvs/lesioncontrol-cuda"
export LESIONCONTROL_SOURCE="$HOME/src/LesionDiffusion-reference"
uv venv --python 3.13 "$LESIONCONTROL_CUDA_ENV"
uv pip install --python "$LESIONCONTROL_CUDA_ENV/bin/python" \
  -r environment/cuda-training.requirements.txt
git clone https://github.com/HengruiTianSJTU/LesionDiffusion.git "$LESIONCONTROL_SOURCE"
git -C "$LESIONCONTROL_SOURCE" checkout --detach \
  880736e2b8a4d9becdb6594da7f8af6db19d04d1
export PYTHONPATH="$LESIONCONTROL_SOURCE/pipeline${PYTHONPATH:+:$PYTHONPATH}"

"$LESIONCONTROL_CUDA_ENV/bin/python" research/train_cuda_adapter.py --help
```

Use the maintained host Node executable with plain pinned Pyright; the
`pyright[nodejs]` extra is not part of this environment. Provision the checker
once, then run the CUDA-specific configuration against that interpreter:

```bash
uv pip install --python "$LESIONCONTROL_CUDA_ENV/bin/python" 'pyright==1.1.414'
"$LESIONCONTROL_CUDA_ENV/bin/pyright" --project research/pyrightconfig.json \
  --pythonpath "$LESIONCONTROL_CUDA_ENV/bin/python" --warnings
```

`make qa` checks the Mac project; the CUDA trainer has the separate configuration
above. The trainer requires CUDA, the external source imports and the original
LINet checkpoint passed through `--weights`. It strictly loads the released EMA
parameters and checks the exported alpha schedule. The checkpoint and optimizer
resume files are trusted PyTorch inputs. Run a preflight before a new campaign:

```bash
"$LESIONCONTROL_CUDA_ENV/bin/python" research/train_cuda_adapter.py \
  --index results/training-bundle/index.json --weights "$LINET_CHECKPOINT" \
  --mode preflight --output results/cuda-preflight

"$LESIONCONTROL_CUDA_ENV/bin/python" research/train_cuda_adapter.py \
  --index results/training-bundle/index.json --weights "$LINET_CHECKPOINT" \
  --mode train --steps 200 --save-every 25 --output results/cuda-training
```

Only adapter parameters are optimized. Resume with `--resume` pointing to a
`step-NNNNNN/training_state.pt`, a new output directory, and the same data,
normalization, seed and learning-rate contract. `--steps` is the target total
step count. Each saved adapter directory contains `metadata.json` and
`adapter.safetensors`; the MLX learned module can check a CUDA preflight fixture
using `--checkpoint`, `--fixture`, `--weights` and `--output`.

## Selection, host preparation and reliability

Conditional selection inputs have schema
`lesioncontrol.conditional_reference_selection.v1`, a `scope` string and `cases`.
Each case supplies `id`, `patient`, `host_patient`, `real_ct`, `real_mask`,
`real_liver_mask`, `clean_ct`, `liver_mask`, `placed_mask` and `known_tumors_mask`.
Reference and host grids are prepared separately; every host input must be free
of known tumor. The exact supplied outline and its 3 mm margin must fit in liver.

`evaluate_adapter.py register` takes `--hosts`, `--bundle`, `--training`,
`--weights` and a new `--root`; subsequent commands are `initial`, `tune`,
`confirm` and `summary`. This script retains the original fixed checkpoint,
gain and step grid. Freeze the selected controls before held-out evaluation.

The held-out preparation scripts retain the original bounded study protocol:

| Input | Required contract |
| --- | --- |
| `prepare_heldout.py --registration` | Schema `lesioncontrol.heldout_reference_registration.v1`; `candidate_patients`, `candidates` containing `patient` and the checked `outline` row, `phase`, and `source_data` unless supplied on the CLI. |
| `--checkpoint-metadata` | Adapter metadata with `train_patients` and `selection_patients`; these memberships are excluded from reference and donor admission. |
| Geometry refinement `--root` | Existing `registration.json`, `real_manifest.json`, initial `manifest.json` and the recorded `crop-bank-extension.json` registration marker. |
| Donor-extension `--registration` | Schema `lesioncontrol.unseen_donor_extension.v1`; `fixed_donor_order`, `donor_metadata`, `excluded_before_selection`, `attempt_only_prior_failed_ids`. Donor metadata rows require `id`, `partition` and `previous_intensity_or_model_result_inspection`. |
| Donor-extension `--reference-manifest` | Existing real-reference manifest with `cases` and the reference image paths/provenance emitted by preparation. |

The donor extension enforces the original twelve-donor/six-reference registration
and named prior-cohort exclusions in its source. Its sorted donor metadata must
match the fixed order. Geometry search checks acquired CT coverage, absence of
native Mass anywhere in a host input, the rigid supplied outline with its liver
margin, and at least 1,000 ring voxels. Preparation emits
`lesioncontrol.conditional_reference_holdout.v1` manifests with the case fields
needed by the reliability runner; it never fills tumors or resizes outlines.

```bash
.venv/bin/python research/run_reliability.py register \
  --root results/reliability --source inputs/prepared_hosts.json \
  --cohort heldout --adapter "$ADAPTER_DIRECTORY" --weights "$MLX_WEIGHTS"
.venv/bin/python research/run_reliability.py run --root results/reliability
.venv/bin/python research/analyze_reliability.py \
  --index results/reliability/index.json --output results/reliability-analysis
```

Registration fixes one lesion per reference patient, seeds 1000–1019, gains zero
and one, model/input file identities and generation controls. The resulting
`lesioncontrol.reliability_runs.v1` index contains `cases`, `seeds` and `draws`;
draws identify patient, lesion, cohort, seed, gain and comparison/diagnostic paths.
The sibling `attempts.jsonl` retains started/completed events, including repeated
resume events. Analysis reads existing feature outputs, requires complete pairs
by default, and reports planned and valid denominators. `--allow-partial` provides
progress summaries. Cohorts `selection`, `heldout` and `previously_evaluated`
remain separate. Seeds are technical repeats; cohort summaries average within
reference patient first. Mean-aligned radiomics shift an analysis copy to the
reference ROI mean, probing attenuation/bin-origin sensitivity while preserving
the saved CT; remaining differences also include histogram and discretization
effects.
