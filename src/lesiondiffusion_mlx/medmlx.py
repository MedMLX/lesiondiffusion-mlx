"""MedMLX file runner for the pinned upstream LMNet and LINet pipeline.

``run`` takes a CT and an organ mask as NIfTI files, standardizes them to a RAS 1 mm
grid, and writes a CT with one synthetic lesion plus the lesion and organ masks. The
lesion shape comes from LMNet, or from a supplied lesion mask that LINet inpaints
directly. The helpers below it cover weight loading, option parsing, geometry, box
placement, and output writing. Research software: outputs are synthetic and need
human review.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict, cast

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MedmlxError,
    ModelExecutionError,
)
from medmlx_core.runtime import import_mlx
from medmlx_core.weights import weights_dir, weights_revision

from lesiondiffusion_mlx._arrays import BoolArray, FloatArray, HostArray, NumericArray
from lesiondiffusion_mlx._nifti import ImageFileError, NiftiImage, nib
from lesiondiffusion_mlx._reference import REFERENCE as _REFERENCE
from lesiondiffusion_mlx._typing import ArrayConstructor

if TYPE_CHECKING:
    from lesiondiffusion_mlx._typing import Array, Dtype, Mlx

from lesiondiffusion_mlx.mlx_model import (
    LINET_CONFIG,
    LMNET_CONFIG,
    BiomedTextEncoder,
    LdmUnet,
    VqAutoencoder,
)
from lesiondiffusion_mlx.mlx_pipeline import (
    LINET_CROP,
    LMNET_CROP,
    encode_attributes,
    encode_source,
    piecewise_ct,
    ras_to_source,
    source_crop,
    source_to_ras,
)
from lesiondiffusion_mlx.mlx_sampling import DdimSampler, lesion_diffusion_timesteps

MODEL_ID: Final = "lesiondiffusion"
DEFAULT_DIFFUSION_STEPS: Final = 200
DEFAULT_SEED: Final = 23
DEFAULT_INFERENCE_MODE: Final = "conditioned"
DEFAULT_OUTPUT_REGION: Final = "full"
# Cosine feather width (mm) for ``output_region="box"``: the generated box is kept,
# blended into the input CT over this margin; farther voxels retain its float32 cast.
LOCALIZATION_MARGIN_MM: Final = 6.0
SOURCE_REF: Final[str] = _REFERENCE["revision"]
CONVERSION_SCHEMA: Final = "medmlx.lesiondiffusion.weights.v1"
CONVERSION_VERSION: Final = 1
WEIGHT_STAGES: Final = ("lmnet", "linet", "vq", "lmnet_text", "linet_text")
SOURCE_INFERENCE_STAGES: Final = tuple(name for name in WEIGHT_STAGES if name != "linet_text")
TOKENIZER_FILE: Final = "tokenizer.json"
CONVERSION_FILE: Final = "conversion.json"
WARNING: Final = (
    "Synthetic lesion output requires human review and is not patient data or ground truth."
)
WEIGHTS_HINT: Final = (
    "Omit weights_path to download MedMLX/lesiondiffusion-mlx automatically, "
    "or supply a folder containing model.safetensors and tokenizer.json."
)

# Canonical labels match upstream organ_type.json and its inference vocabulary.
SOURCE_ORGAN_LABELS: Final[dict[str, tuple[int, str]]] = {
    name.lower().replace(" ", "_"): (int(label), name)
    for label, name in _REFERENCE["organ_labels"].items()
}
# Hollow organs for box placement, as in upstream preprocessing. The cavity phrase in
# _lmnet_attributes follows upstream's inference code, whose set also has gallbladder.
_HOLLOW_ORGANS = frozenset({"colon", "stomach", "esophagus", "urinary_bladder"})
_CAVITY_TEXT = {
    "inside_parenchyma": "within the parenchymal organ",
    "protruding_from_parenchyma": "protruding from the parenchymal organ",
    "within_lumen": "within the lumen of a hollow organ",
    "wall_thickening": "wall thickening",
}

_SYNTHETIC_CT = "synthetic_ct.nii.gz"
_RAW_SYNTHETIC_CT = "raw_synthetic_ct.nii.gz"
_LESION_MASK = "lesion_mask.nii.gz"
_ORGAN_MASK = "organ_mask.nii.gz"

MODEL: Final[dict[str, str]] = {
    "id": MODEL_ID,
    "label": "LesionDiffusion CT lesion insertion",
    "task": "generation",
    "run": "lesiondiffusion_mlx.medmlx:run",
    "summary": (
        "Insert one phenotype-controlled synthetic lesion into a CT, given a binary organ "
        "mask, or inpaint a supplied binary lesion mask. Outputs use a RAS 1 mm grid."
    ),
}


class NativeRuntime(TypedDict):
    """Runtime facts recorded under ``details.generation_spec.native_runtime``.

    Covers device, per-stage compute dtypes, the inference mode and the arithmetic it
    selects, the text attributes sent to each network, per-stage records, and a
    plain-language statement of what a seed reproduces. ``lesion_mask_source`` is
    ``"lmnet"`` or ``"supplied"``; a supplied mask skips LMNet, so its attempts are 0,
    its conditioning is empty and its stage record has ``status="skipped"``.
    """

    device: str
    dtype: dict[str, str]
    inference_mode: str
    convolution_arithmetic: str
    lesion_mask_source: str
    lmnet_attempts: int
    lmnet_conditioning: dict[str, object]
    lmnet_anatomy_context: str
    linet_conditioning: dict[str, object]
    linet_skip_context: bool
    lmnet_stage: dict[str, object]
    linet_stage: dict[str, object]
    model_organ_label: int
    sampling: str
    synchronization: str
    seed_contract: str


class OutputValidation(TypedDict):
    """Checks applied to the written outputs; ``lesion_voxel_count`` is on the output grid."""

    schema_version: int
    lesion_voxel_count: int
    finite: bool
    method: str


class GenerationSpec(TypedDict):
    """Provenance of one lesion: box (voxel indices on the standardized grid), placement,
    runtime facts and output checks.

    ``anatomy_mask`` is present only when an anatomy map was supplied,
    ``lesion_mask`` only when a lesion mask was supplied (its file, voxel counts and
    the box derived from it; see ``prepare_lesion_mask``), and ``compositing`` only
    for ``output_region="box"`` (see ``localize_output``).
    """

    schema_version: str
    source_ref: str
    bbox_center_voxel: list[int]
    bbox_size_voxel: list[int]
    placement: str
    output_mapping: str
    resampling: str
    native_runtime: NativeRuntime
    output_validation: OutputValidation
    anatomy_mask: NotRequired[object]
    lesion_mask: NotRequired[object]
    compositing: NotRequired[dict[str, object]]


class RunDetails(TypedDict):
    """JSON-safe record of one run; ``elapsed_s`` is seconds, ``started_at`` UTC ISO 8601."""

    model_id: str
    source_ref: str
    organ: str
    lesion_type: str
    seed: int
    diffusion_steps: int
    generation_spec: GenerationSpec
    stages: dict[str, dict[str, object]]
    output_region: str
    output_kind_by_role: dict[str, str]
    output_shape_by_role: dict[str, list[int]]
    output_dtype_by_role: dict[str, str]
    output_labels_by_role: dict[str, list[dict[str, str | int]]]
    weights_path: str
    weights_revision: str | None
    started_at: str
    elapsed_s: float
    requires_human_review: bool
    warning: str


class RunResult(TypedDict):
    """Return value of ``run``.

    ``outputs`` maps a role to a POSIX file path: ``synthetic_image``, ``target_mask``
    and ``organ_mask``, plus ``raw_synthetic_image`` when ``output_region="box"``.
    """

    outputs: dict[str, str]
    details: RunDetails


type _PathArg = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class LesionDiffusionOptions:
    """Validated controls for one run.

    ``organ`` is a key of ``SOURCE_ORGAN_LABELS``. ``bbox_size_mm`` is the box extent
    (X, Y, Z) in mm along the standardized RAS axes, at least 2 mm each, or ``None``
    when a supplied lesion mask defines the box (see ``prepare_geometry``).
    ``bbox_center_voxel`` is an optional (I, J, K) voxel index on that grid; when it is
    ``None`` the box is placed from ``seed``. ``placement`` picks the placement rule and
    the cavity phrase in the text. ``shape`` through ``specific_features`` are free
    text for the upstream report. ``seed`` (0 to 2**32 - 1) seeds MT19937 box placement
    and MLX sampling. ``inference_mode`` is ``"conditioned"`` or ``"source"`` and
    ``output_region`` is ``"full"`` or ``"box"``; see ``run``.

    Raises:
        InvalidInputError: From construction, for any value outside these contracts.
    """

    organ: str
    lesion_type: str
    bbox_size_mm: tuple[float, float, float] | None
    placement: str
    shape: str = "irregular"
    density: str = "hypodense lesion"
    density_variation: str = "heterogeneous"
    enhancement: str = "non-contrast CT"
    surface: str = "ill-defined margin"
    adjacent_relationship: str = "no close relationship with surrounding organs"
    specific_features: str = ""
    bbox_center_voxel: tuple[int, int, int] | None = None
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS
    seed: int = DEFAULT_SEED
    inference_mode: str = DEFAULT_INFERENCE_MODE
    output_region: str = DEFAULT_OUTPUT_REGION

    def __post_init__(self) -> None:
        """Validate inference, geometry, and lesion-generation options."""
        if self.inference_mode not in {"conditioned", "source"}:
            raise InvalidInputError("LesionDiffusion inference_mode must be conditioned or source")
        if self.output_region not in {"full", "box"}:
            raise InvalidInputError("LesionDiffusion output_region must be full or box")
        if self.organ not in SOURCE_ORGAN_LABELS:
            raise InvalidInputError(f"Unknown upstream organ: {self.organ}")
        if not self.lesion_type.strip():
            raise InvalidInputError("LesionDiffusion lesion_type must not be empty")
        if self.bbox_size_mm is not None:
            if len(self.bbox_size_mm) != 3:
                raise InvalidInputError("LesionDiffusion bbox_size_mm must contain X, Y, Z")
            for value in self.bbox_size_mm:
                if not math.isfinite(value):
                    raise InvalidInputError("LesionDiffusion bbox_size_mm must be finite")
                if value < 2:
                    raise InvalidInputError("LesionDiffusion bbox dimensions must be at least 2 mm")
        if self.placement not in _CAVITY_TEXT:
            raise InvalidInputError(f"Unknown LesionDiffusion placement: {self.placement}")
        for name, value in (
            ("shape", self.shape),
            ("density_variation", self.density_variation),
            ("density", self.density),
            ("enhancement", self.enhancement),
            ("surface", self.surface),
            ("adjacent_relationship", self.adjacent_relationship),
        ):
            if not value.strip():
                raise InvalidInputError(f"LesionDiffusion {name} must not be empty")
        if self.bbox_center_voxel is not None:
            if len(self.bbox_center_voxel) != 3:
                raise InvalidInputError("LesionDiffusion bbox_center_voxel must contain I, J, K")
            if any(not _is_integer(value) for value in self.bbox_center_voxel):
                raise InvalidInputError("LesionDiffusion bbox_center_voxel values must be integers")
            if any(value < 0 for value in self.bbox_center_voxel):
                raise InvalidInputError("LesionDiffusion bbox_center_voxel must not be negative")
        if not _is_integer(self.diffusion_steps):
            raise InvalidInputError("LesionDiffusion diffusion_steps must be an integer")
        try:
            lesion_diffusion_timesteps(self.diffusion_steps)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        if not _is_integer(self.seed):
            raise InvalidInputError("LesionDiffusion seed must be an integer")
        if self.seed < 0:
            raise InvalidInputError("LesionDiffusion seed must not be negative")
        if self.seed >= 2**32:
            raise InvalidInputError("LesionDiffusion seed must be less than 2**32 for MT19937")


@dataclass(frozen=True, slots=True)
class _Geometry:
    """Standardized inputs shared by every stage.

    ``reference`` and the arrays are on the even-padded RAS 1 mm grid the networks use;
    ``output_reference`` is the unpadded standardized grid that outputs are cropped
    back to. ``bbox_mask`` is uint8 with 2 inside the box. ``center`` is the box
    center (I, J, K) and ``bbox_size_voxel`` its nominal size; the box spans
    ``size // 2`` voxels on each side of the center. ``lesion_mask`` is the supplied
    lesion that replaces LMNet, or ``None`` when LMNet draws it.
    """

    reference: NiftiImage
    output_reference: NiftiImage
    source_data: NumericArray
    organ_mask: BoolArray
    organ_label: int
    organ_source: dict[str, object]
    bbox_mask: HostArray
    bbox_size_voxel: tuple[int, int, int]
    center: tuple[int, int, int]
    anatomy_mask: HostArray | None = None
    lesion_mask: BoolArray | None = None


def run(
    *,
    image_path: _PathArg,
    organ_mask_path: _PathArg,
    anatomy_mask_path: _PathArg | None = None,
    lesion_mask_path: _PathArg | None = None,
    output_dir: _PathArg,
    weights_path: _PathArg | None = None,
    organ: str,
    lesion_type: str,
    bbox_size_mm: Sequence[float] | None = None,
    placement: str,
    shape: str = "irregular",
    density: str = "hypodense lesion",
    density_variation: str = "heterogeneous",
    enhancement: str = "non-contrast CT",
    surface: str = "ill-defined margin",
    adjacent_relationship: str = "no close relationship with surrounding organs",
    specific_features: str = "",
    bbox_center_voxel: Sequence[int] | None = None,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    seed: int = DEFAULT_SEED,
    inference_mode: str = DEFAULT_INFERENCE_MODE,
    output_region: str = DEFAULT_OUTPUT_REGION,
    overwrite: bool = False,
) -> RunResult:
    """Insert one synthetic lesion into a CT and write NIfTI outputs.

    LMNet generates a lesion mask conditioned on a box, then LINet inpaints the CT latent
    under that mask and the VQ decoder reconstructs the volume (upstream's v2 inpainting
    route). A supplied ``lesion_mask_path`` replaces LMNet's mask: LMNet does not run,
    and LINet paints a new lesion with that shape and the requested phenotype text.
    CT intensities are taken as Hounsfield units. The CT and masks must be
    axis-aligned grids; they are reoriented and resampled to RAS 1 mm (linear for the
    CT, nearest for masks) and padded to even extents for the VQ stages, then outputs
    are cropped back to that standardized grid, which is the input grid when the
    input is already RAS 1 mm. An LMNet lesion is not constrained to the box or the
    organ. Output is synthetic and needs human review.

    Args:
        image_path: 3-D CT, ``.nii`` or ``.nii.gz``, finite values in HU.
        organ_mask_path: Binary organ mask (0 background, 1 organ) on the same grid
            after standardization.
        anatomy_mask_path: Optional multi-organ map with upstream TotalSegmentator v1
            labels 0-104 on the same grid; its label for ``organ`` must equal the
            binary mask. Without it LMNet sees only the target organ. It conditions
            only LMNet, so it is rejected with ``lesion_mask_path``.
        lesion_mask_path: Optional binary lesion mask (0 background, 1 lesion) on the
            same grid after standardization, used instead of LMNet's mask. It must be
            one face-connected lesion that overlaps the organ mask, keeps a voxel on
            LINet's half-resolution latent grid and fits LINet's inpainting crop
            (``LINET_CROP``, 128 voxels per axis). The box is derived from its bounds
            (see ``lesion_box``), so ``bbox_size_mm`` and ``bbox_center_voxel`` must be
            omitted.
        output_dir: Directory for the outputs, created if needed.
        weights_path: Optional folder containing ``model.safetensors`` and
            ``tokenizer.json``. Omit to download ``MedMLX/lesiondiffusion-mlx``.
            Weights use stage-prefixed keys; ``source`` mode does not require
            ``linet_text``. Conversion receipts are not required.
        organ: Key of ``SOURCE_ORGAN_LABELS`` (lower-case name, spaces as underscores).
        lesion_type: Free text such as ``"tumor"``. A selected LMNet component of at
            most 400 voxels (50 if the text contains ``"stone"``) triggers one resample.
            With ``lesion_mask_path`` it only names the output label.
        bbox_size_mm: Box extent (X, Y, Z) in mm, each at least 2. Required unless
            ``lesion_mask_path`` is supplied, and rejected with it.
        placement: ``inside_parenchyma``, ``protruding_from_parenchyma``,
            ``within_lumen`` or ``wall_thickening``. It guides automatic box placement
            and, in ``conditioned`` mode, LINet's text. LMNet's cavity phrase is fixed
            by organ and lesion type, as in the pinned inference path. With
            ``lesion_mask_path`` only the LINet text remains; it is not checked
            against the mask.
        shape, density, density_variation, enhancement, surface,
            adjacent_relationship, specific_features: Free-text phenotype for the
            upstream report; empty strings are rejected except ``specific_features``.
        bbox_center_voxel: Optional (I, J, K) box center on the standardized grid;
            rejected with ``lesion_mask_path``.
        diffusion_steps: DDIM steps for both stages; must give a valid upstream
            uniform schedule (see ``lesion_diffusion_timesteps``).
        seed: Seeds box placement and native MLX sampling. MLX's RNG differs from
            Torch's, so a seed repeats within MLX only.
        inference_mode: ``"conditioned"`` uses native MLX FP16 convolution and the
            trained LINet text path, with resampling at output-affine coordinates.
            ``"source"`` reproduces upstream's disabled image text path, Torch CPU
            reduction order and legacy ``zoom`` resampling coordinates.
        output_region: ``"full"`` writes upstream's full-volume reconstruction, whose HU
            are clipped to [-1000, 1000] and can differ from the input outside the
            lesion. ``"box"`` keeps the generated CT over the box, widened to the
            lesion's bounding box when the lesion extends past it, feathers that region
            into the input CT over ``LOCALIZATION_MARGIN_MM``, and writes the full
            reconstruction as ``raw_synthetic_image``. Outside the blend region it
            retains the standardized input cast to float32, which can round
            higher-precision input. The lesion mask is not clipped; every lesion voxel
            lies in the kept region.
        overwrite: Replace existing output files instead of refusing. A failure before
            the final file moves leaves no new outputs; a failure during those moves
            can leave partial outputs that require ``overwrite`` for a retry.

    Returns:
        ``{"outputs": ..., "details": ...}``. ``synthetic_image`` is float32 HU;
        ``target_mask`` and ``organ_mask`` are uint8 with value 1 for lesion and
        organ. All keep the standardized grid's affine, qform and sform.

    Raises:
        InvalidInputError: For invalid or conflicting options, unreadable or mismatched
            inputs, a lesion mask LINet cannot inpaint as one lesion, or outputs that
            would overwrite an input or an existing file.
        AssetNotReadyError: If the weights directory is missing or not current.
        ModelExecutionError: If a stage produces non-finite values, LMNet yields an
            empty lesion mask, or another error occurs during execution.
    """

    options = _parse_options(
        organ=organ,
        lesion_type=lesion_type,
        bbox_size_mm=bbox_size_mm,
        placement=placement,
        shape=shape,
        density=density,
        density_variation=density_variation,
        enhancement=enhancement,
        surface=surface,
        adjacent_relationship=adjacent_relationship,
        specific_features=specific_features,
        bbox_center_voxel=bbox_center_voxel,
        diffusion_steps=diffusion_steps,
        seed=seed,
        inference_mode=inference_mode,
        output_region=output_region,
        lesion_mask_supplied=lesion_mask_path is not None,
    )
    source, supplied_mask, outputs = _run_paths(
        image_path=image_path,
        organ_mask_path=organ_mask_path,
        output_dir=output_dir,
        output_region=options.output_region,
        overwrite=overwrite,
    )
    anatomy_path = _optional_input_path(anatomy_mask_path, "anatomy mask", outputs)
    lesion_path = _optional_input_path(lesion_mask_path, "lesion mask", outputs)
    if anatomy_path is not None and lesion_path is not None:
        raise InvalidInputError(
            "LesionDiffusion anatomy_mask_path conditions only LMNet, which "
            "lesion_mask_path skips; omit anatomy_mask_path"
        )
    folder = weights_dir(MODEL_ID) if weights_path is None else weights_path
    directory = require_weights(folder, inference_mode=options.inference_mode)
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        geometry = prepare_geometry(
            source=source,
            supplied_mask=supplied_mask,
            supplied_anatomy=anatomy_path,
            supplied_lesion=lesion_path,
            options=options,
        )
        spec, stages = _execute_run(geometry, options, directory, outputs)
    except MedmlxError:
        raise
    except Exception as exc:
        raise ModelExecutionError(f"LesionDiffusion failed during execution: {exc}") from exc
    return _run_result(options, geometry, outputs, directory, spec, stages, started_at, started)


def _run_paths(
    *,
    image_path: _PathArg,
    organ_mask_path: _PathArg,
    output_dir: _PathArg,
    output_region: str,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Path]]:
    """Resolve input and output paths and prevent output collisions."""
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be a boolean")
    source = Path(image_path).expanduser().resolve()
    supplied_mask = Path(organ_mask_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise InvalidInputError(f"Missing input CT NIfTI: {source}")
    if not supplied_mask.is_file():
        raise InvalidInputError(f"Missing organ mask NIfTI: {supplied_mask}")
    outputs = output_paths(output, output_region)
    if source in outputs.values():
        raise InvalidInputError("LesionDiffusion output files must not overwrite the input CT")
    if supplied_mask in outputs.values():
        raise InvalidInputError(
            "LesionDiffusion output files must not overwrite the input organ mask"
        )
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"LesionDiffusion output path is not a directory: {output}")
    if not overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise InvalidInputError(
                f"{existing[0]} already exists; pass overwrite=true to replace outputs"
            )
    return source, supplied_mask, outputs


def _optional_input_path(
    path: _PathArg | None,
    role: str,
    outputs: Mapping[str, Path],
) -> Path | None:
    """Resolve an optional input file and refuse an output that would overwrite it."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if resolved in outputs.values():
        raise InvalidInputError(f"LesionDiffusion output files must not overwrite the {role}")
    return resolved


def _execute_run(
    geometry: _Geometry,
    options: LesionDiffusionOptions,
    directory: Path,
    outputs: dict[str, Path],
) -> tuple[GenerationSpec, dict[str, dict[str, object]]]:
    """Run native generation and write its validated outputs."""
    model_label, _ = source_organ_label(options.organ)
    report = _structured_report(
        reference=geometry.reference,
        organ_mask=geometry.organ_mask,
        organ_label=model_label,
        center=geometry.center,
        bbox_size=geometry.bbox_size_voxel,
        options=options,
    )
    attributes = cast(dict[str, object], cast(list[object], report[str(model_label)])[2])
    mx = cast("Mlx", import_mlx())
    stages: dict[str, dict[str, object]] = {}
    image, mask, facts = run_native_pipeline(
        source_ct_ras=geometry.source_data,
        organ_mask_ras=geometry.organ_mask,
        bbox_mask_ras=geometry.bbox_mask,
        options=options,
        attributes=attributes,
        weights_dir=directory,
        mx=mx,
        on_stage_completed=stages.__setitem__,
        anatomy_mask_ras=geometry.anatomy_mask,
        lesion_mask_ras=geometry.lesion_mask,
    )
    validation, compositing = materialize_arrays(
        outputs=outputs,
        geometry=geometry,
        image_data=image,
        lesion_data=mask,
        output_region=options.output_region,
    )
    spec: GenerationSpec = {
        "schema_version": "medmlx.lesiondiffusion.generation.v1",
        "source_ref": SOURCE_REF,
        "bbox_center_voxel": list(geometry.center),
        "bbox_size_voxel": list(geometry.bbox_size_voxel),
        "placement": options.placement,
        "output_mapping": "source_las_zyx_to_input_ras_grid",
        "resampling": "source_zoom" if options.inference_mode == "source" else "affine_ras_1mm",
        "native_runtime": facts,
        "output_validation": validation,
    }
    if geometry.anatomy_mask is not None:
        spec["anatomy_mask"] = geometry.organ_source["anatomy_mask"]
    if geometry.lesion_mask is not None:
        spec["lesion_mask"] = geometry.organ_source["lesion_mask"]
    if compositing is not None:
        spec["compositing"] = compositing
    return spec, stages


def _run_result(
    options: LesionDiffusionOptions,
    geometry: _Geometry,
    outputs: dict[str, Path],
    directory: Path,
    spec: GenerationSpec,
    stages: dict[str, dict[str, object]],
    started_at: datetime,
    started: float,
) -> RunResult:
    """Build the output paths and generation metadata for a completed run."""
    grid = [int(value) for value in geometry.output_reference.shape]
    details: RunDetails = {
        "model_id": MODEL_ID,
        "source_ref": SOURCE_REF,
        "organ": options.organ,
        "lesion_type": options.lesion_type,
        "seed": options.seed,
        "diffusion_steps": options.diffusion_steps,
        "generation_spec": spec,
        "stages": stages,
        "output_region": options.output_region,
        "output_kind_by_role": {
            "synthetic_image": "synthetic_image",
            "target_mask": "label_map",
            "organ_mask": "label_map",
            **(
                {"raw_synthetic_image": "synthetic_image"}
                if "raw_synthetic_image" in outputs
                else {}
            ),
        },
        "output_shape_by_role": {role: grid for role in outputs},
        "output_dtype_by_role": {
            "synthetic_image": "float32",
            "target_mask": "uint8",
            "organ_mask": "uint8",
            **({"raw_synthetic_image": "float32"} if "raw_synthetic_image" in outputs else {}),
        },
        "output_labels_by_role": {
            "target_mask": [{"name": options.lesion_type, "value": 1}],
            "organ_mask": [{"name": options.organ, "value": geometry.organ_label}],
        },
        "weights_path": directory.as_posix(),
        "weights_revision": weights_revision(directory),
        "started_at": started_at.isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - started, 4),
        "requires_human_review": True,
        "warning": WARNING,
    }
    return {
        "outputs": {role: path.as_posix() for role, path in outputs.items()},
        "details": cast(RunDetails, _json_safe(details)),
    }


def run_native_pipeline(
    *,
    source_ct_ras: NumericArray,
    organ_mask_ras: BoolArray,
    bbox_mask_ras: HostArray,
    options: LesionDiffusionOptions,
    attributes: Mapping[str, object],
    weights_dir: Path,
    mx: Mlx,
    on_stage_completed: Callable[[str, dict[str, object]], None],
    anatomy_mask_ras: HostArray | None = None,
    lesion_mask_ras: BoolArray | None = None,
) -> tuple[HostArray, HostArray, NativeRuntime]:
    """Run LMNet, then LINet and the VQ decoder, with the released EMA weights.

    Volumes are in standardized RAS ``(X, Y, Z)`` order on the even-padded grid and are
    converted to upstream's ``(Z, Y, X)`` layout internally. Seeds MLX's RNG with
    ``options.seed``. Each stage is evaluated and exported to the host before the next
    loads, to keep peak memory down.

    A supplied ``lesion_mask_ras`` takes the place of LMNet's selected component, as
    upstream LINet inpaints whichever label file its list names: LMNet is not loaded
    and its stage record says ``skipped``. LINet still receives the phenotype text.

    Args:
        source_ct_ras: CT in HU.
        organ_mask_ras: Boolean target-organ mask.
        bbox_mask_ras: Box volume, 2 inside the box and 0 elsewhere (``bbox_mask``).
        options: Validated run options.
        attributes: Upstream report fields from ``_structured_report``.
        weights_dir: Converted-weights directory (see ``require_weights``).
        mx: The imported ``mlx.core`` module.
        on_stage_completed: Called as ``on_stage_completed("lmnet" | "linet", record)``
            after each stage with its settings and timings.
        anatomy_mask_ras: Optional TotalSegmentator v1 label map (0-104); defaults to
            the target organ carrying its upstream label id. LMNet-only, so it cannot
            be combined with ``lesion_mask_ras``.
        lesion_mask_ras: Optional boolean lesion mask validated by
            ``prepare_lesion_mask``; when given, LMNet is skipped.

    Returns:
        ``(generated_ct, label_map, facts)``. ``generated_ct`` is the full
        VQ-reconstructed CT, float32 HU clipped to [-1000, 1000]. ``label_map`` is
        uint8 with 2 for lesion, 1 for the remaining organ voxels and 0 elsewhere.
        ``facts`` records the runtime settings.

    Raises:
        ModelExecutionError: If a stage produces non-finite values or LMNet yields an
            empty lesion.
        InvalidInputError: If the grid is smaller than a network crop, or both an
            anatomy map and a lesion mask are given.
    """
    if anatomy_mask_ras is not None and lesion_mask_ras is not None:
        raise InvalidInputError(
            "LesionDiffusion anatomy context conditions only LMNet, which a supplied "
            "lesion mask skips"
        )
    directory = weights_dir
    tokenizer_path = directory / TOKENIZER_FILE
    mx.random.seed(options.seed)
    image = ras_to_source(source_ct_ras).astype(np.float32)
    organ = ras_to_source(organ_mask_ras).astype(bool)
    organ_id, _ = source_organ_label(options.organ)
    if lesion_mask_ras is None:
        bbox = ras_to_source(bbox_mask_ras) == 2
        anatomy = (
            ras_to_source(anatomy_mask_ras)
            if anatomy_mask_ras is not None
            else organ.astype(np.uint8) * organ_id
        )
        lm_attributes = _lmnet_attributes(attributes, options)
        lesion, lm_stage = _run_lmnet(
            organ, bbox, options, lm_attributes, directory, tokenizer_path, mx, anatomy=anatomy
        )
    else:
        lm_attributes: dict[str, object] = {}
        lesion = ras_to_source(lesion_mask_ras).astype(bool)
        lm_stage: dict[str, object] = {"status": "skipped", "reason": "supplied_lesion_mask"}
    on_stage_completed("lmnet", lm_stage)
    lin_attributes = _linet_attributes(attributes, options)
    generated_ras, lin_stage = _run_linet(
        image, lesion, options, lin_attributes, directory, tokenizer_path, mx
    )
    on_stage_completed("linet", lin_stage)
    lesion_ras = source_to_ras(np.where(lesion, 2, organ.astype(np.uint8)))
    details: NativeRuntime = {
        "device": str(mx.default_device()),
        "dtype": {"lmnet": "float32", "linet": "float16", "vq": "float16"},
        "inference_mode": options.inference_mode,
        "convolution_arithmetic": (
            "torch_cpu_source_reduction" if options.inference_mode == "source" else "mlx_native"
        ),
        "lesion_mask_source": "lmnet" if lesion_mask_ras is None else "supplied",
        "lmnet_attempts": cast(int, lm_stage.get("attempts", 0)),
        "lmnet_conditioning": lm_attributes,
        "lmnet_anatomy_context": (
            "lmnet_skipped"
            if lesion_mask_ras is not None
            else "supplied_source_labels"
            if anatomy_mask_ras is not None
            else "target_organ_only"
        ),
        "linet_conditioning": lin_attributes,
        "linet_skip_context": options.inference_mode == "source",
        "lmnet_stage": lm_stage,
        "linet_stage": lin_stage,
        "model_organ_label": organ_id,
        "sampling": "source_uniform_ddim_eta_1",
        "synchronization": "mx.eval_each_ddim_step_and_export",
        "seed_contract": (
            "MLX native RNG; same seed is reproducible within MLX, not identical to Torch RNG"
        ),
    }
    return generated_ras, lesion_ras, details


def _linet_attributes(
    attributes: Mapping[str, object], options: LesionDiffusionOptions
) -> dict[str, object]:
    """Build image-synthesis text attributes for the selected organ and mode."""
    _, organ_name = source_organ_label(options.organ)
    omitted = {"size"}
    if options.inference_mode == "conditioned":
        # The report's numeric CT value is the input organ mean, not a requested
        # lesion density. It must not override the user's categorical phenotype.
        omitted.add("CT value")
    return {
        **{
            key: value
            for key, value in attributes.items()
            if key not in omitted and (options.inference_mode == "source" or value != "")
        },
        "lesion location": organ_name,
    }


def _lmnet_attributes(
    attributes: Mapping[str, object],
    options: LesionDiffusionOptions,
) -> dict[str, object]:
    """Build mask-synthesis text attributes using the pinned source conventions."""
    _, organ_name = source_organ_label(options.organ)
    lm_attributes = dict(attributes)
    lm_attributes["lesion location"] = organ_name
    # The mask generator is not given size, density or the organ-mean CT sentence.
    for key in ("size", "CT value", "density variations", "density"):
        lm_attributes.pop(key, None)
    if not lm_attributes.get("specific features"):
        lm_attributes.pop("specific features", None)
    # This matches the pinned inference path: its supplied coarseg and target
    # are identical, so solid organs receive the protruding condition. The
    # ``placement`` option does not enter here; it only shapes box placement and
    # LINet's text. Gallbladder counts as hollow here but not for box placement.
    lm_attributes["cavity"] = (
        (
            "within the lumen of a hollow organ,"
            if "stone" in options.lesion_type
            else "wall thickening,"
        )
        if options.organ in {"colon", "urinary_bladder", "esophagus", "gallbladder", "stomach"}
        else "protruding from the parenchymal organ,"
    )
    return lm_attributes


def _run_lmnet(
    organ: BoolArray,
    bbox: BoolArray,
    options: LesionDiffusionOptions,
    lm_attributes: Mapping[str, object],
    directory: Path,
    tokenizer_path: Path,
    mx: Mlx,
    *,
    anatomy: HostArray,
) -> tuple[BoolArray, dict[str, object]]:
    """Generate and validate a lesion mask within the selected crop."""
    crop = source_crop(bbox, LMNET_CROP)
    lm_started = time.perf_counter()
    optimized = options.inference_mode == "conditioned"
    lmnet = LdmUnet(
        load_stage(directory, "lmnet", mx, dtype=mx.float32),
        mx,
        LMNET_CONFIG,
        fused_attention=optimized,
        native_kernels=optimized,
    )
    text = BiomedTextEncoder(load_stage(directory, "lmnet_text", mx, dtype=mx.float32), mx)
    condition = encode_attributes(lm_attributes, text, tokenizer_path, mx)
    denoiser = mx.compile(lmnet.__call__) if optimized else lmnet.__call__
    # Conditioning channels: anatomy label id / 104 (the upstream label count),
    # target organ, box; shape (1, 3, *LMNET_CROP).
    conditioning = cast(ArrayConstructor, mx.array)(
        np.stack((anatomy[crop].astype(np.float64) / 104.0, organ[crop], bbox[crop]))[None],
        dtype=mx.float32,
    )
    sampler = DdimSampler(lmnet.weights["alphas_cumprod"], mx, options.diffusion_steps)
    lesion = np.zeros(bbox.shape, dtype=bool)
    attempts = 0
    samples: Array | None = None
    # Volume QC: resample once if the selected component is too small (at most 400
    # voxels, 50 for stones); the second attempt is kept either way.
    for attempt in range(1, 3):
        attempts = attempt
        samples = sampler(denoiser, condition, conditioning)
        mx.eval(samples)
        if not bool(mx.all(mx.isfinite(samples)).item()):
            raise ModelExecutionError("Native LMNet produced non-finite samples")
        component = select_lesion_component(cast(FloatArray, np.array(samples))[0, 0], bbox[crop])
        lesion[crop] = component
        if int(np.sum(component)) > (50 if "stone" in options.lesion_type else 400):
            break
    # Release LMNet before LINet and the VQ autoencoder load, to bound peak memory.
    del lmnet, denoiser, text, condition, conditioning, sampler, samples
    gc.collect()
    mx.clear_cache()
    if not np.any(lesion):
        raise ModelExecutionError(
            "LesionDiffusion LMNet QC rejected this seed: mask has no lesion label 2"
        )
    lm_stage: dict[str, object] = {
        "status": "completed",
        "device": str(mx.default_device()),
        "dtype": "float32",
        "crop_shape": list(LMNET_CROP),
        "diffusion_steps_actual": len(lesion_diffusion_timesteps(options.diffusion_steps)),
        "attempts": attempts,
        "lesion_voxels": int(np.sum(lesion)),
        "finite": True,
        "attention": "flash_fp32_self_mlx_fused_padded_cross" if optimized else "source_explicit",
        "denoiser_compiled": optimized,
        "synchronized_elapsed_s": time.perf_counter() - lm_started,
    }
    return lesion, lm_stage


def _run_linet(
    image: FloatArray,
    lesion: BoolArray,
    options: LesionDiffusionOptions,
    lin_attributes: Mapping[str, object],
    directory: Path,
    tokenizer_path: Path,
    mx: Mlx,
) -> tuple[HostArray, dict[str, object]]:
    """Generate the lesion image using the selected mask and attributes."""
    lin_started = time.perf_counter()
    source_arithmetic = options.inference_mode == "source"
    linet = LdmUnet(
        load_stage(directory, "linet", mx, dtype=mx.float16),
        mx,
        replace(LINET_CONFIG, skip_context=source_arithmetic),
        source_arithmetic=source_arithmetic,
        fused_attention=not source_arithmetic,
    )
    vq = VqAutoencoder(
        load_stage(directory, "vq", mx, dtype=mx.float16),
        mx,
        source_arithmetic=source_arithmetic,
    )
    nax_convolutions = len(linet.nax_weights) + len(vq.nax_weights)
    condition: Array | None = None
    if not source_arithmetic:
        text = BiomedTextEncoder(load_stage(directory, "linet_text", mx, dtype=mx.float32), mx)
        condition = encode_attributes(lin_attributes, text, tokenizer_path, mx)
        del text
    # Force the lazy weight loads here so load_and_condition_s measures them.
    mx.eval(*linet.weights.values(), *vq.weights.values())
    load_elapsed = time.perf_counter() - lin_started
    encode_started = time.perf_counter()
    native_image = piecewise_ct(cast(ArrayConstructor, mx.array)(image[None, None]), mx).astype(
        mx.float16
    )
    latent = encode_source(vq, native_image, mx)
    if not bool(mx.all(mx.isfinite(latent)).item()):
        raise ModelExecutionError("Native VQ encoder produced non-finite latents")
    encode_elapsed = time.perf_counter() - encode_started
    # Lesion mask on the latent grid: nearest-neighbour resize, then binarize.
    mask = vq.resize_nearest(
        cast(ArrayConstructor, mx.array)(lesion[None, None], dtype=mx.float16),
        tuple(int(value) for value in latent.shape[2:]),
    )
    mask = (mask > 0.5).astype(mx.float16)
    selection = (slice(None), slice(None), *_linet_latent_window(lesion))
    sampler = DdimSampler(linet.weights["alphas_cumprod"], mx, options.diffusion_steps)
    denoiser = linet.__call__ if source_arithmetic else mx.compile(linet.__call__)
    mx.eval(mask)
    sampling_started = time.perf_counter()
    generated = sampler(denoiser, condition, latent[selection], mask=mask[selection])
    latent[selection] = generated.astype(mx.float16)
    mx.eval(latent)
    if not bool(mx.all(mx.isfinite(latent)).item()):
        raise ModelExecutionError("Native LINet produced non-finite latents before quantization")
    sampling_elapsed = time.perf_counter() - sampling_started
    # Free the UNet before decoding; only the VQ autoencoder and latent are still needed.
    del linet, denoiser, sampler, generated, native_image, mask, condition
    gc.collect()
    mx.clear_cache()
    decode_started = time.perf_counter()
    decoded = vq.decode(latent)
    mx.eval(decoded)
    if not bool(mx.all(mx.isfinite(decoded)).item()):
        raise ModelExecutionError(
            "Native VQ decoder produced non-finite voxels before CT remapping"
        )
    decoded = piecewise_ct(decoded.astype(mx.float32), mx, inverse=True)
    mx.eval(decoded)
    if not bool(mx.all(mx.isfinite(decoded)).item()):
        raise ModelExecutionError("Native LINet/VQ produced non-finite CT voxels")
    generated_ras = source_to_ras(np.array(decoded)[0, 0]).astype(np.float32)
    decode_elapsed = time.perf_counter() - decode_started
    lin_stage: dict[str, object] = {
        "status": "completed",
        "device": str(mx.default_device()),
        "dtype": "float16",
        "crop_shape": list(LINET_CROP),
        "latent_shape": list(latent.shape),
        "output_shape": list(image.shape),
        "diffusion_steps_actual": len(lesion_diffusion_timesteps(options.diffusion_steps)),
        "finite": True,
        "attention": "source_explicit" if source_arithmetic else "mlx_fused_padded_fp16",
        "denoiser_compiled": not source_arithmetic,
        "nax_convolutions": nax_convolutions,
        "load_and_condition_s": load_elapsed,
        "encode_s": encode_elapsed,
        "sampling_s": sampling_elapsed,
        "decode_and_export_s": decode_elapsed,
        "synchronized_elapsed_s": time.perf_counter() - lin_started,
    }
    return generated_ras, lin_stage


def _linet_latent_window(lesion: BoolArray) -> tuple[slice, slice, slice]:
    """Return LINet's sampling window on the latent grid for a source-order lesion.

    The image crop is ``LINET_CROP`` centred on the lesion's bounding box. The latent
    has half the image resolution (VQ factor two), so the crop bounds are halved,
    rounding down, as upstream does.
    """
    crop = source_crop(lesion, LINET_CROP)
    return (
        slice(int(crop[0].start) // 2, int(crop[0].stop) // 2),
        slice(int(crop[1].start) // 2, int(crop[1].stop) // 2),
        slice(int(crop[2].start) // 2, int(crop[2].stop) // 2),
    )


def _require_linet_coverage(lesion: BoolArray) -> None:
    """Refuse a source-order lesion that LINet would leave partly or wholly unpainted.

    On the even-padded grid, ``_run_linet``'s nearest resize to the half-size latent
    reads every even source index, so LINet sees ``lesion[::2, ::2, ::2]``, and it
    samples only the window from ``_linet_latent_window``. The written lesion mask is
    the full-resolution mask, so a site missing from either would be labeled lesion
    without being inpainted.
    """
    latent = lesion[::2, ::2, ::2]
    if not np.any(latent):
        raise InvalidInputError(
            "Lesion mask has no voxel on LINet's half-resolution latent grid; "
            "supply a lesion at least two voxels thick on each axis"
        )
    window = _linet_latent_window(lesion)
    if np.count_nonzero(latent[window]) != np.count_nonzero(latent):
        raise InvalidInputError(
            "Lesion mask must fit LINet's inpainting crop of "
            f"{LINET_CROP[::-1]} voxels (RAS X, Y, Z)"
        )


def select_lesion_component(samples: NumericArray, reference: BoolArray) -> BoolArray:
    """Pick the lesion mask from LMNet samples as upstream does.

    For each threshold in 0.9, 0.8, ..., 0.3 the samples are binarized (``>``), reduced
    to their largest connected component (SciPy's default face connectivity), and the
    threshold whose component volume is closest to the box volume wins; ties keep
    the higher threshold.

    Args:
        samples: LMNet output crop ``(D, H, W)`` in source order.
        reference: Boolean box mask of the same shape; only its voxel count is used.

    Returns:
        Boolean lesion mask with the shape of ``samples``.

    If every threshold produces an empty component, returns an empty mask; the
    runner handles that outcome after its retry.
    """
    from scipy import ndimage

    best: BoolArray | None = None
    difference = float("inf")
    for threshold in np.arange(0.9, 0.2, -0.1):
        mask = samples > threshold
        labels, count = ndimage.label(mask)
        if count:
            sizes = ndimage.sum(mask, labels, range(count + 1))
            mask = labels == int(np.argmax(sizes[1:]) + 1)
        current = abs(int(np.sum(mask)) - int(np.sum(reference)))
        if current < difference:
            difference, best = current, mask.copy()
    if best is None:
        raise ModelExecutionError("LesionDiffusion lesion selection found no threshold")
    return best


def source_organ_label(organ: str) -> tuple[int, str]:
    """Return ``(upstream label id, report name)`` for an organ key of ``SOURCE_ORGAN_LABELS``."""

    return SOURCE_ORGAN_LABELS[organ]


# --- converted weights ----------------------------------------------------------------


def require_weights(path: _PathArg, *, inference_mode: str = DEFAULT_INFERENCE_MODE) -> Path:
    """Validate the native directory and required safetensors stage prefixes.

    Source mode does not need the LINet text encoder. This checks the file header,
    not every parameter's contents. Conversion receipts are not consulted.
    """
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise AssetNotReadyError(
            f"LesionDiffusion MLX weights directory is missing: {directory}",
            reason=f"missing {directory.as_posix()}",
            hint=WEIGHTS_HINT,
        )
    packed = directory / "model.safetensors"
    for file in (packed, directory / TOKENIZER_FILE):
        if not file.is_file():
            raise AssetNotReadyError(
                f"Native MLX LesionDiffusion weights are not ready: missing {file}",
                reason=f"Missing native LesionDiffusion asset: {file}",
                hint=WEIGHTS_HINT,
            )
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(packed, framework="np") as tensors:
            names = tensors.keys()
    except (SafetensorError, OSError) as exc:
        raise AssetNotReadyError(
            f"Native MLX LesionDiffusion weights are not readable: {packed}: {exc}",
            reason=f"Unreadable {packed.name}",
            hint=WEIGHTS_HINT,
        ) from exc
    present = {key.split(".", 1)[0] for key in names}
    stages = SOURCE_INFERENCE_STAGES if inference_mode == "source" else WEIGHT_STAGES
    for name in stages:
        if name not in present:
            raise AssetNotReadyError(
                f"Native MLX LesionDiffusion weights are missing stage {name}",
                reason=f"Missing stage in {packed.name}",
                hint=WEIGHTS_HINT,
            )
    return directory


def load_conversion(directory: Path) -> dict[str, object]:
    """Read and check ``conversion.json``; return its provenance fields.

    Raises:
        ValueError: If the file is not a JSON object, or its schema, source revision,
            EMA flag, layout or precision profile is not the current one.
        OSError: If the file cannot be read.
    """
    metadata = _load_json_object(directory / CONVERSION_FILE)
    if (
        metadata.get("schema_version") != CONVERSION_SCHEMA
        or metadata.get("source_ref") != SOURCE_REF
    ):
        raise ValueError("LesionDiffusion MLX conversion schema/source is not current")
    if metadata.get("converter_version") != CONVERSION_VERSION or metadata.get("ema") is not True:
        raise ValueError("LesionDiffusion MLX conversion must use current EMA conversion")
    if metadata.get("layout") != "source_torch_ncdhw" or metadata.get("stored_dtype") != "float32":
        raise ValueError("LesionDiffusion MLX conversion layout/dtype is not current")
    if metadata.get("inference_dtype") != {
        "lmnet": "float32",
        "linet": "float16",
        "vq": "float16",
    }:
        raise ValueError("LesionDiffusion MLX conversion precision profile is not current")
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "converter_version",
            "source_ref",
            "layout",
            "stored_dtype",
            "inference_dtype",
            "ema",
        )
    }


def load_stage(directory: Path, name: str, mx: Mlx, *, dtype: Dtype) -> dict[str, Array]:
    """Load one stage from the native safetensors bundle and cast its arrays.

    Removes the ``<stage>.`` prefix. Conversion receipts are not required.
    """
    loaded = mx.load(str(directory / "model.safetensors"), format="safetensors")
    prefix = name + "."
    return {
        key.removeprefix(prefix): value.astype(dtype)
        for key, value in loaded.items()
        if key.startswith(prefix)
    }


def _load_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object or report an invalid metadata file."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], payload)


# --- options --------------------------------------------------------------------------


def _parse_options(
    *,
    organ: object,
    lesion_type: object,
    bbox_size_mm: object,
    placement: object,
    shape: object,
    density: object,
    density_variation: object,
    enhancement: object,
    surface: object,
    adjacent_relationship: object,
    specific_features: object,
    bbox_center_voxel: object,
    diffusion_steps: object,
    seed: object,
    inference_mode: object,
    output_region: object = DEFAULT_OUTPUT_REGION,
    lesion_mask_supplied: bool = False,
) -> LesionDiffusionOptions:
    """Validate request values and construct generation options."""
    for name, value in (
        ("organ", organ),
        ("lesion_type", lesion_type),
        ("placement", placement),
        ("shape", shape),
        ("density", density),
        ("density_variation", density_variation),
        ("enhancement", enhancement),
        ("surface", surface),
        ("adjacent_relationship", adjacent_relationship),
        ("specific_features", specific_features),
        ("inference_mode", inference_mode),
        ("output_region", output_region),
    ):
        if not isinstance(value, str):
            raise InvalidInputError(f"LesionDiffusion {name} must be a string")
    _check_box_source(
        bbox_size_mm=bbox_size_mm,
        bbox_center_voxel=bbox_center_voxel,
        lesion_mask_supplied=lesion_mask_supplied,
    )
    size: tuple[float, float, float] | None = None
    if bbox_size_mm is not None:
        if isinstance(bbox_size_mm, str | bytes) or not isinstance(bbox_size_mm, Sequence):
            raise InvalidInputError("LesionDiffusion bbox_size_mm must contain X, Y, Z")
        size_values = cast(Sequence[object], bbox_size_mm)
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) for value in size_values
        ):
            raise InvalidInputError("LesionDiffusion bbox_size_mm values must be numbers")
        size = cast(
            tuple[float, float, float],
            tuple(float(value) for value in cast(Sequence[int | float], size_values)),
        )
    center: tuple[int, int, int] | None = None
    if bbox_center_voxel is not None:
        if isinstance(bbox_center_voxel, str | bytes) or not isinstance(
            bbox_center_voxel, Sequence
        ):
            raise InvalidInputError("LesionDiffusion bbox_center_voxel must contain I, J, K")
        center = cast(tuple[int, int, int], tuple(cast(Sequence[object], bbox_center_voxel)))
    return LesionDiffusionOptions(
        organ=cast(str, organ),
        lesion_type=cast(str, lesion_type),
        bbox_size_mm=size,
        placement=cast(str, placement),
        shape=cast(str, shape),
        density=cast(str, density),
        density_variation=cast(str, density_variation),
        enhancement=cast(str, enhancement),
        surface=cast(str, surface),
        adjacent_relationship=cast(str, adjacent_relationship),
        specific_features=cast(str, specific_features),
        bbox_center_voxel=center,
        diffusion_steps=cast(int, diffusion_steps),
        seed=cast(int, seed),
        inference_mode=cast(str, inference_mode),
        output_region=cast(str, output_region),
    )


def _check_box_source(
    *,
    bbox_size_mm: object,
    bbox_center_voxel: object,
    lesion_mask_supplied: bool,
) -> None:
    """Require one source for the box: ``bbox_size_mm`` or a supplied lesion mask.

    A supplied lesion mask defines the box (see ``lesion_box``), so a requested size or
    center would contradict it and is refused rather than ignored.
    """
    if not lesion_mask_supplied:
        if bbox_size_mm is None:
            raise InvalidInputError(
                "LesionDiffusion bbox_size_mm is required unless lesion_mask_path is supplied"
            )
        return
    for name, value in (("bbox_size_mm", bbox_size_mm), ("bbox_center_voxel", bbox_center_voxel)):
        if value is not None:
            raise InvalidInputError(
                f"LesionDiffusion derives the box from lesion_mask_path; omit {name}"
            )


def _is_integer(value: object) -> bool:
    """Recognize integer inputs while excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


# --- geometry -------------------------------------------------------------------------


def prepare_geometry(
    *,
    source: Path,
    supplied_mask: Path,
    options: LesionDiffusionOptions,
    supplied_anatomy: Path | None = None,
    supplied_lesion: Path | None = None,
) -> _Geometry:
    """Standardize the inputs and place the box.

    Loads the CT onto the RAS 1 mm grid, edge-pads to even extents, loads the organ
    mask (and optional anatomy map) onto the same grid, converts ``bbox_size_mm`` to
    voxels with ``round`` at the grid spacing, and resolves the box center. Conditioned
    mode uses affine-coordinate resampling; source mode retains upstream ``zoom``.

    With ``supplied_lesion`` the lesion mask is loaded instead (``prepare_lesion_mask``)
    and the box is derived from it (``lesion_box``); ``options`` must then carry no
    box size or center. LMNet does not run, so only LINet's crop must fit the volume.

    Raises:
        InvalidInputError: If a file is unreadable, a grid is oblique or does not
            match, a mask has invalid labels, the anatomy map disagrees with the organ
            mask or uses labels above 104, the box options conflict with the lesion
            mask, the box does not fit, a network's fixed crop is larger than the
            padded standardized volume, or a supplied lesion mask is not one lesion
            LINet can inpaint.
    """
    _check_box_source(
        bbox_size_mm=options.bbox_size_mm,
        bbox_center_voxel=options.bbox_center_voxel,
        lesion_mask_supplied=supplied_lesion is not None,
    )
    resampling: Literal["source", "affine"] = (
        "source" if options.inference_mode == "source" else "affine"
    )
    output_reference, standardized = _load_standardized_ct(source, resampling=resampling)
    reference, source_data = _pad_to_even(output_reference, standardized, mode="edge")
    organ_label = 1
    organ_mask, organ_source = prepare_organ_mask(
        reference=reference,
        supplied_mask=supplied_mask,
        resampling=resampling,
    )
    anatomy_mask = None
    if supplied_anatomy is not None:
        anatomy_mask = _load_label_map(
            reference, supplied_anatomy, "anatomy mask", max_label=104, resampling=resampling
        )
        values = np.unique(anatomy_mask)
        source_label, _ = source_organ_label(options.organ)
        if not np.array_equal(anatomy_mask == source_label, organ_mask):
            raise InvalidInputError(
                "Anatomy mask selected-organ label must match the binary organ mask"
            )
        anatomy_mask = anatomy_mask.astype(np.uint8)
        organ_source["anatomy_mask"] = {
            "path": supplied_anatomy.as_posix(),
            "label_vocabulary": "upstream_totalsegmentator_v1_104",
            "labels_present": [int(value) for value in values],
        }
    lesion_mask = None
    if supplied_lesion is None:
        # _check_box_source guarantees a size when no lesion mask is supplied.
        size_mm = cast(tuple[float, float, float], options.bbox_size_mm)
        spacing_mm = tuple(float(value) for value in reference.header.get_zooms()[:3])
        bbox_size = tuple(
            int(round(size_mm[index] / spacing_mm[index]))  # noqa: RUF046 - verbatim
            for index in range(3)
        )
        bbox_size_voxel = cast(tuple[int, int, int], bbox_size)
        center = resolve_bbox_center(
            organ_mask=organ_mask,
            bbox_size=bbox_size_voxel,
            placement=options.placement,
            requested_center=options.bbox_center_voxel,
            seed=options.seed,
            organ=options.organ,
        )
        bbox_array = bbox_mask(reference.shape, center, bbox_size_voxel)
        _require_network_crops(reference.shape, (("LMNet", LMNET_CROP), ("LINet", LINET_CROP)))
    else:
        # LMNet does not run; LINet's crop must fit before the lesion is checked against it.
        _require_network_crops(reference.shape, (("LINet", LINET_CROP),))
        lesion_mask, lesion_source = prepare_lesion_mask(
            reference=reference,
            supplied_mask=supplied_lesion,
            organ_mask=organ_mask,
            resampling=resampling,
        )
        center, bbox_size_voxel = lesion_box(lesion_mask)
        organ_source["lesion_mask"] = {
            **lesion_source,
            "box": "lesion_bounding_box_widened_to_even_voxel_indices",
        }
        bbox_array = bbox_mask(reference.shape, center, bbox_size_voxel)
    return _Geometry(
        reference,
        output_reference,
        source_data,
        organ_mask,
        organ_label,
        organ_source,
        bbox_array,
        bbox_size_voxel,
        center,
        anatomy_mask,
        lesion_mask,
    )


def _require_network_crops(
    shape: tuple[int, ...],
    crops: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    """Reject a padded RAS grid smaller than a network's fixed crop, before it loads.

    ``crops`` pairs a stage name with its crop constant, in source Z, Y, X order.
    """
    for stage, crop in crops:
        required = crop[::-1]
        if any(size < minimum for size, minimum in zip(shape, required, strict=True)):
            raise InvalidInputError(
                f"Pinned LesionDiffusion {stage} crop requires RAS shape >= {required}; got {shape}"
            )


_RAS_AXCODES: Final = ("R", "A", "S")


def _is_ras_1mm_axis_aligned(image: NiftiImage) -> bool:
    """True when ``image`` is already RAS, 1 mm isotropic, and axis-aligned."""
    orientation = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    return bool(
        orientation == _RAS_AXCODES
        and np.allclose(spacing, (1.0, 1.0, 1.0), atol=0.01, rtol=0)
        and np.allclose(image.affine[:3, :3], np.eye(3), atol=1e-4, rtol=0)
    )


def _standardize_to_ras_1mm(
    image: NiftiImage,
    data: NumericArray,
    *,
    order: Literal[0, 1],
    resampling: Literal["source", "affine"] = "source",
) -> tuple[NiftiImage, NumericArray]:
    """Reorient and resample to an axis-aligned RAS 1 mm grid.

    Spatial units are normalized to millimeters, including affine translations
    and coded forms. An unspecified unit retains the established mm interpretation.
    Reorientation (axis permutation and sign flips) is exact; isotropic resampling
    is linear for ``order=1`` CT and nearest for ``order=0`` labels. ``source`` keeps
    upstream's ``zoom`` convention, whose sample coordinates can differ from the
    assigned output affine. ``affine`` samples at the world coordinates of that
    affine, using edge extension for CT and zero background for labels. Both keep
    the reoriented origin and round(size * spacing) extents, then callers pad odd
    extents. Integer CT becomes float32 when interpolated in ``affine`` mode.
    Already-conforming input is returned unchanged.
    A grid that is still not axis-aligned after reorientation (a true oblique or
    sheared affine) is rejected rather than resampled; this port does not implement
    general affine resampling.
    """
    try:
        spatial_unit, temporal_unit = image.header.get_xyzt_units()
        scale = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}[spatial_unit]
    except KeyError as error:
        raise InvalidInputError(
            "LesionDiffusion NIfTI units must be mm, meter, micron or unknown"
        ) from error
    if spatial_unit != "mm":
        affine_mm = np.array(image.affine, dtype=np.float64, copy=True)
        affine_mm[:3] *= scale
        header = nib.Nifti1Header.from_header(image.header)
        header.set_xyzt_units("mm", temporal_unit)
        normalized = nib.Nifti1Image(cast(HostArray, data), affine_mm, header)
        for read_form, write_form in (
            (image.get_qform, normalized.set_qform),
            (image.get_sform, normalized.set_sform),
        ):
            form, code = read_form(coded=True)
            form_mm = np.array(affine_mm if form is None else form, copy=True)
            if form is not None:
                form_mm[:3] *= scale
            write_form(form_mm, code=int(code))
        image = normalized
    if _is_ras_1mm_axis_aligned(image):
        return image, data
    affine = np.asarray(image.affine, dtype=np.float64)
    current = nib.io_orientation(affine)
    target = nib.orientations.axcodes2ornt(_RAS_AXCODES)
    transform = nib.orientations.ornt_transform(current, target)
    reoriented = cast(
        NumericArray, nib.orientations.apply_orientation(cast(HostArray, data), transform)
    )
    reoriented_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    linear = reoriented_affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diagonal(linear)), atol=1e-4, rtol=0):
        raise InvalidInputError(
            "LesionDiffusion requires an axis-aligned grid; this NIfTI's transform is "
            "oblique and is not resampled automatically"
        )
    zoom_factors = tuple(float(value) for value in np.abs(np.diagonal(linear)))
    if np.allclose(zoom_factors, 1.0, atol=1e-6):
        resampled, new_affine = reoriented, reoriented_affine
    else:
        new_affine = np.array(reoriented_affine, dtype=np.float64, copy=True)
        new_affine[:3, :3] = np.diag(np.sign(np.diagonal(linear)))
        if resampling == "source":
            from scipy.ndimage import zoom

            resampled = cast(NumericArray, zoom(reoriented, zoom_factors, order=order))
        else:
            from scipy.ndimage import affine_transform

            output_shape = tuple(
                max(1, round(size * spacing))
                for size, spacing in zip(reoriented.shape, zoom_factors, strict=True)
            )
            output_dtype = (
                np.dtype(np.float32)
                if order == 1 and np.issubdtype(reoriented.dtype, np.integer)
                else reoriented.dtype
            )
            resampled = cast(
                NumericArray,
                affine_transform(
                    reoriented,
                    np.linalg.solve(reoriented_affine, new_affine),
                    output_shape=output_shape,
                    output=output_dtype,
                    order=order,
                    mode="nearest" if order == 1 else "constant",
                    cval=0,
                    prefilter=False,
                ),
            )
    header = nib.Nifti1Header.from_header(image.header)
    header.set_data_dtype(resampled.dtype.type)
    standardized = nib.Nifti1Image(cast(HostArray, resampled), new_affine, header)
    return standardized, resampled


def _pad_to_even(
    image: NiftiImage, data: NumericArray, *, mode: Literal["edge", "constant"]
) -> tuple[NiftiImage, NumericArray]:
    """Pad trailing voxels so every spatial axis is even, keeping the origin fixed.

    Resampling to 1 mm can land on an odd extent; the full-volume VQ stages need
    even dimensions. ``edge`` repeats the border voxel (for an intensity volume),
    ``constant`` zero-pads (for a label mask already handled as background 0).
    """
    pad_width = tuple((0, 1) if size % 2 else (0, 0) for size in data.shape)
    if not any(width[1] for width in pad_width):
        return image, data
    padded = (
        np.pad(data, pad_width, mode="edge")
        if mode == "edge"
        else np.pad(data, pad_width, mode="constant", constant_values=0)
    )
    header = nib.Nifti1Header.from_header(image.header)
    header.set_data_dtype(padded.dtype.type)
    padded_image = nib.Nifti1Image(cast(HostArray, padded), image.affine, header)
    return padded_image, padded


def load_reference_nifti(
    path: Path, *, resampling: Literal["source", "affine"] = "source"
) -> tuple[NiftiImage, NumericArray]:
    """Load CT on an even-padded RAS 1 mm grid; default to pinned source resampling.

    ``resampling="affine"`` selects the world-coordinate sampling used by the
    conditioned runner. See ``_standardize_to_ras_1mm`` for the edge convention.
    """
    image, data = _load_standardized_ct(path, resampling=resampling)
    return _pad_to_even(image, data, mode="edge")


def _load_standardized_ct(
    path: Path, *, resampling: Literal["source", "affine"] = "source"
) -> tuple[NiftiImage, NumericArray]:
    """Load a finite 3-D NIfTI CT and standardize it to the RAS 1 mm grid (unpadded)."""
    if not path.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError("LesionDiffusion input must be a .nii or .nii.gz NIfTI")
    try:
        image = nib.load(str(path))
    except (OSError, ValueError, ImageFileError) as exc:
        raise InvalidInputError(f"Could not read input CT NIfTI: {path}") from exc
    if len(image.shape) != 3:
        raise InvalidInputError(f"LesionDiffusion input must be 3D; got {image.shape}")
    if not np.all(np.isfinite(image.affine)):
        raise InvalidInputError("LesionDiffusion input affine contains non-finite values")
    data = cast(NumericArray, np.asanyarray(image.dataobj))
    if not np.all(np.isfinite(data)):
        raise InvalidInputError("LesionDiffusion input CT contains non-finite values")
    return _standardize_to_ras_1mm(image, data, order=1, resampling=resampling)


def prepare_organ_mask(
    *,
    reference: NiftiImage,
    supplied_mask: Path,
    resampling: Literal["source", "affine"] = "source",
) -> tuple[BoolArray, dict[str, object]]:
    """Load the binary organ mask on the CT's even-padded grid.

    ``resampling`` must match the CT's selected convention.

    Returns:
        ``(mask, provenance)``: a boolean array on ``reference``'s grid and a record of
        the mask file.

    Raises:
        InvalidInputError: If the file is unreadable, the grid differs from
            ``reference``, labels are negative or fractional, or the foreground is
            not exactly the single label 1 (this also rejects an empty mask).
    """
    provenance: dict[str, object] = {
        "producer": "supplied_organ_mask",
        "mask": supplied_mask.as_posix(),
        "binary_label_id": 1,
        "requires_human_review": True,
        "is_ground_truth": False,
    }
    data = _load_label_map(
        reference, supplied_mask, "organ mask", max_label=1, resampling=resampling
    )
    positive = np.unique(data[data > 0])
    if len(positive) != 1:
        raise InvalidInputError(
            "Organ mask must contain exactly one foreground label; multi-label maps are ambiguous"
        )
    return data > 0, provenance


def prepare_lesion_mask(
    *,
    reference: NiftiImage,
    supplied_mask: Path,
    organ_mask: BoolArray,
    resampling: Literal["source", "affine"] = "source",
) -> tuple[BoolArray, dict[str, object]]:
    """Load a supplied binary lesion mask that replaces LMNet's output.

    The file uses the runner's ``target_mask`` encoding, 0 background and 1 lesion, and
    is standardized like the organ mask (RAS 1 mm, nearest neighbour, zero padding to
    even extents) onto the CT's grid. Upstream LINet reads lesion label 2 of LMNet's
    label map and ignores its organ label 1; this runner hands LINet the boolean
    lesion directly, so no organ label is needed and the lesion may extend past the
    organ. It must overlap the organ, whose name LINet's text gives as the location.

    The runner inpaints one lesion in one ``LINET_CROP`` window of LINet's
    half-resolution latent, the case upstream reaches with a single connected
    component. The mask must therefore be one face-connected component (SciPy's
    default, as in upstream's component selection), and every lesion site on the
    latent grid must fall inside that window; otherwise part of the written lesion
    would be left unpainted.

    ``resampling`` must match the CT's selected convention.

    Returns:
        ``(mask, provenance)``: a boolean array on ``reference``'s grid and a record of
        the file and its voxel counts on that grid.

    Raises:
        InvalidInputError: If the file is unreadable, not binary or on another grid, or
            the standardized lesion is empty, has several components, misses the
            organ, has no voxel on LINet's latent grid or exceeds LINet's crop.
    """
    from scipy import ndimage

    data = _load_label_map(
        reference, supplied_mask, "lesion mask", max_label=1, resampling=resampling
    )
    lesion = data > 0
    if not np.any(lesion):
        raise InvalidInputError(
            "Lesion mask contains no lesion voxels (label 1) after standardization to RAS 1 mm"
        )
    _, components = ndimage.label(lesion)
    if components != 1:
        raise InvalidInputError(
            "Lesion mask must be one connected lesion; found "
            f"{components} components after standardization to RAS 1 mm"
        )
    voxels_inside_organ = int(np.count_nonzero(lesion & organ_mask))
    if voxels_inside_organ == 0:
        raise InvalidInputError(
            "Lesion mask does not overlap the organ mask; place the lesion in the selected organ"
        )
    _require_linet_coverage(ras_to_source(lesion).astype(bool))
    provenance: dict[str, object] = {
        "producer": "supplied_lesion_mask",
        "mask": supplied_mask.as_posix(),
        "binary_label_id": 1,
        "voxel_count": int(np.count_nonzero(lesion)),
        "voxels_inside_organ": voxels_inside_organ,
    }
    return lesion, provenance


# Label-range errors name how to obtain an accepted map for each role.
_LABEL_RANGE_ERRORS: Final = {
    "organ mask": (
        "Supplied organ mask must be binary with label 1 for the selected organ; "
        "extract that organ from a labeled anatomy map first"
    ),
    "anatomy mask": "Anatomy mask must use upstream TotalSegmentator v1 labels 0-104",
    "lesion mask": (
        "Supplied lesion mask must be binary with label 1 for the lesion; "
        "extract one lesion from a labeled segmentation first"
    ),
}


def _load_label_map(
    reference: NiftiImage,
    supplied_mask: Path,
    role: str,
    *,
    max_label: Literal[1, 104],
    resampling: Literal["source", "affine"] = "source",
) -> NumericArray:
    """Load an integer label map onto ``reference``'s grid (nearest-neighbour, zero-padded).

    ``role`` names the map in error messages. Raises ``InvalidInputError`` if the file
    is unreadable, has non-finite values or labels outside the integer range
    ``0..max_label``, or its grid differs from ``reference`` after standardization.
    Validate source labels before resampling, which can otherwise hide invalid voxels.
    """
    if not supplied_mask.is_file():
        raise InvalidInputError(f"Missing {role} NIfTI: {supplied_mask}")
    try:
        image = nib.load(str(supplied_mask))
    except (OSError, ValueError, ImageFileError) as exc:
        raise InvalidInputError(f"Could not read {role} NIfTI: {supplied_mask}") from exc
    data = cast(NumericArray, np.asanyarray(image.dataobj))
    if not np.all(np.isfinite(data)):
        raise InvalidInputError(f"LesionDiffusion {role} contains non-finite values")
    values = np.unique(data)
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise InvalidInputError(f"{role.capitalize()} labels must be non-negative integers")
    if np.any(values > max_label):
        raise InvalidInputError(_LABEL_RANGE_ERRORS[role])
    image, data = _standardize_to_ras_1mm(image, data, order=0, resampling=resampling)
    image, data = _pad_to_even(image, data, mode="constant")
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise InvalidInputError(f"LesionDiffusion {role} must share the input CT grid")
    return data


def resolve_bbox_center(
    *,
    organ_mask: BoolArray,
    bbox_size: tuple[int, int, int],
    placement: str,
    requested_center: tuple[int, int, int] | None,
    seed: int,
    organ: str = "liver",
) -> tuple[int, int, int]:
    """Return the box center ``(I, J, K)`` in voxels, as upstream preprocessing places it.

    An explicit ``requested_center`` is returned unchanged if its box fits the grid; it
    need not lie inside the organ. Otherwise a ``np.random.RandomState(seed)`` (MT19937)
    picks a candidate organ voxel. For a solid organ with ``inside_parenchyma`` the
    candidates are shuffled and the first whose whole box lies in the organ is used.
    For a hollow organ, ``within_lumen`` draws from the organ eroded by 5 voxels and
    ``wall_thickening`` from the organ minus that erosion (the whole organ if the
    region is empty). Any other case, or no fitting candidate, draws one voxel at
    random. The center is finally clamped so the box stays inside the grid.

    Raises:
        InvalidInputError: If a requested box does not fit, or the organ has no voxels.
    """
    from scipy.ndimage import binary_erosion

    if requested_center is not None:
        if bbox_slices(requested_center, bbox_size, organ_mask.shape) is None:
            raise InvalidInputError("Requested bbox_center_voxel does not fit the image grid")
        return requested_center
    rng = np.random.RandomState(seed)
    region = organ_mask
    if organ in _HOLLOW_ORGANS and placement in {"within_lumen", "wall_thickening"}:
        eroded = binary_erosion(organ_mask, iterations=5)
        region = eroded if placement == "within_lumen" else organ_mask & ~eroded
        if not np.any(region):
            region = organ_mask
    candidates = np.argwhere(region)
    if not len(candidates):
        raise InvalidInputError("No organ voxels available for box placement")
    half = np.asarray(bbox_size) // 2
    center = None
    if organ not in _HOLLOW_ORGANS and placement == "inside_parenchyma":
        rng.shuffle(candidates)
        for candidate in candidates:
            low, high = candidate - half, candidate + half
            if np.any(low < 0) or np.any(high >= organ_mask.shape):
                continue
            if np.all(organ_mask[tuple(slice(a, b + 1) for a, b in zip(low, high, strict=True))]):
                center = candidate
                break
    if center is None:
        # Upstream rebuilds the unshuffled candidates after an unsuccessful scan.
        candidates = np.argwhere(region)
        center = candidates[rng.randint(len(candidates))]
    center = np.maximum(center, half)
    center = np.minimum(center, np.asarray(organ_mask.shape) - half - 1)
    return int(center[0]), int(center[1]), int(center[2])


def bbox_slices(
    center: tuple[int, int, int],
    size: tuple[int, int, int],
    shape: tuple[int, ...],
) -> tuple[slice, slice, slice] | None:
    """Return the box as slices ``[center - size // 2, center + size // 2)``, or ``None``.

    Each side has length ``2 * (size // 2)``, so an odd size loses one voxel. ``None``
    means the box falls outside ``shape``.
    """
    half = np.asarray(size) // 2
    low, high = np.asarray(center) - half, np.asarray(center) + half
    if np.any(low < 0) or np.any(high > shape):
        return None
    slices = [slice(int(a), int(b)) for a, b in zip(low, high, strict=True)]
    return slices[0], slices[1], slices[2]


def bbox_mask(
    shape: tuple[int, ...],
    center: tuple[int, int, int],
    size: tuple[int, int, int],
) -> HostArray:
    """Return a uint8 volume of ``shape`` with 2 inside the box and 0 elsewhere.

    Raises:
        InvalidInputError: If the box extends outside ``shape``.
    """
    slices = bbox_slices(center, size, shape)
    if slices is None:
        raise InvalidInputError("Resolved LesionDiffusion bbox extends outside the image")
    mask = np.zeros(shape, dtype=np.uint8)
    mask[slices] = 2
    return mask


def lesion_box(lesion: BoolArray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(center, size)`` in voxels of the box a supplied lesion mask defines.

    The box is the lesion's bounding box widened outward to even voxel indices. It is
    therefore made of whole cells of LINet's VQ latent grid (two voxels per axis on the
    even-padded grid) and contains every latent cell LINet inpaints for this lesion.
    Its even extent is represented exactly by ``bbox_slices`` as ``center ± size // 2``
    and always fits the even-padded grid. For ``output_region="box"`` it is the region
    kept generated; the cosine feather is the only margin beyond it.

    Args:
        lesion: Non-empty boolean lesion mask on the even-padded RAS grid.
    """
    points = np.argwhere(lesion)
    low = points.min(axis=0) // 2 * 2
    high = (points.max(axis=0) + 2) // 2 * 2
    size = high - low
    center = low + size // 2
    return (
        (int(center[0]), int(center[1]), int(center[2])),
        (int(size[0]), int(size[1]), int(size[2])),
    )


def _structured_report(
    *,
    reference: NiftiImage,
    organ_mask: BoolArray,
    organ_label: int,
    center: tuple[int, int, int],
    bbox_size: tuple[int, int, int],
    options: LesionDiffusionOptions,
) -> dict[str, object]:
    """Build upstream's structured report for one lesion.

    Returns ``{label: [shape, "", fields, [[center], [size]]]}``. Phrases end with a
    comma, as upstream's do. ``lesion location`` starts as the numeric label and is
    replaced by the organ name before encoding. ``size`` is in voxels. ``CT value``
    states the mean HU of the organ in the standardized CT, although its wording says
    "lesion region"; it is not a lesion density and is never embedded as text.
    """
    image_data = cast(NumericArray, np.asanyarray(reference.dataobj))
    organ_values = image_data[organ_mask]
    mean_hu = float(np.mean(organ_values)) if organ_values.size else 0.0
    details = {
        "enhancement status": _with_comma(options.enhancement),
        "lesion location": str(organ_label),
        "shape": _with_comma(options.shape),
        "size": f"[[{' '.join(str(value) for value in bbox_size)}]] pixel,",
        "density": _with_comma(options.density),
        "density variations": _with_comma(options.density_variation),
        "CT value": f"Mean density of the lesion region is approximately {mean_hu:.1f} HU,",
        "surface characteristics": _with_comma(options.surface),
        "relationship with adjacent organs": _with_comma(options.adjacent_relationship),
        "specific features": _with_comma(options.specific_features)
        if options.specific_features
        else "",
        "cavity": _with_comma(_CAVITY_TEXT[options.placement]),
    }
    return {
        str(organ_label): [
            _with_comma(options.shape),
            "",
            details,
            [[[int(value) for value in center]], [[int(value) for value in bbox_size]]],
        ]
    }


# --- outputs --------------------------------------------------------------------------


def materialize_arrays(
    *,
    outputs: Mapping[str, Path],
    geometry: _Geometry,
    image_data: HostArray,
    lesion_data: HostArray,
    output_region: str = DEFAULT_OUTPUT_REGION,
) -> tuple[OutputValidation, dict[str, object] | None]:
    """Validate the generated arrays and write every output NIfTI file.

    All files, including ``organ_mask``, are first written to hidden temporary files
    beside their destinations and moved into place only after every write succeeds.
    A failure before the final moves leaves no new output that would block a retry
    without ``overwrite``; a failure or interruption during those moves can leave a subset.

    In ``box`` mode the generated CT is kept over the smallest box containing both
    the requested box and the lesion (which LMNet does not confine to the box), so
    every voxel of ``target_mask`` lies on generated intensities. For a supplied
    lesion mask that is the derived box itself.

    Args:
        outputs: Role to destination path, from ``output_paths``.
        geometry: Standardized inputs; ``output_reference`` defines the written grid.
        image_data: Generated CT (HU) on the even-padded grid.
        lesion_data: Label map on the same grid; 2 marks the lesion.
        output_region: ``"full"`` or ``"box"``; see ``run``.

    Returns:
        ``(output_validation, compositing)``; ``compositing`` is ``None`` unless
        ``output_region="box"``.

    Raises:
        ModelExecutionError: If shapes differ from the grid, values are non-finite, or
            the label map has no lesion.
    """
    if (
        image_data.shape != geometry.reference.shape
        or lesion_data.shape != geometry.reference.shape
    ):
        raise ModelExecutionError("LesionDiffusion generated arrays differ from the input grid")
    if not np.all(np.isfinite(image_data)) or not np.all(np.isfinite(lesion_data)):
        raise ModelExecutionError("LesionDiffusion generated arrays contain non-finite voxels")
    lesion = (lesion_data[_input_grid(geometry.output_reference)] == 2).astype(np.uint8)
    if not np.any(lesion):
        raise ModelExecutionError("LesionDiffusion mask output does not contain lesion label 2")
    images: list[tuple[HostArray, str]] = [(image_data, "synthetic_image")]
    compositing: dict[str, object] | None = None
    if output_region == "box":
        box = cast(
            tuple[slice, slice, slice],
            bbox_slices(geometry.center, geometry.bbox_size_voxel, geometry.reference.shape),
        )
        # Bounding voxels of the written lesion; ``lesion`` is non-empty here.
        points = np.argwhere(lesion)
        low, high = points.min(axis=0), points.max(axis=0) + 1
        kept = cast(
            tuple[slice, slice, slice],
            tuple(
                slice(min(int(part.start), int(a)), max(int(part.stop), int(b)))
                for part, a, b in zip(box, low, high, strict=True)
            ),
        )
        localized, compositing = localize_output(
            original=geometry.source_data,
            generated=cast(NumericArray, image_data),
            bbox=kept,
            spacing_mm=cast(
                tuple[float, float, float],
                tuple(float(value) for value in geometry.reference.header.get_zooms()[:3]),
            ),
        )
        # A supplied lesion mask's box already contains the lesion (see ``lesion_box``).
        compositing["kept_region"] = (
            "lesion_mask_box"
            if geometry.lesion_mask is not None
            else "requested_box_union_lesion_bounding_box"
        )
        compositing["requested_box"] = {
            "start_voxel": [int(part.start) for part in box],
            "stop_voxel_exclusive": [int(part.stop) for part in box],
        }
        images = [(localized, "synthetic_image"), (image_data, "raw_synthetic_image")]
    organ = np.where(geometry.organ_mask, geometry.organ_label, 0).astype(np.uint8)
    writes: list[tuple[HostArray, str, type[np.generic]]] = [
        *((data, role, np.float32) for data, role in images),
        (lesion, "target_mask", np.uint8),
        (organ, "organ_mask", np.uint8),
    ]
    token = f".lesiondiffusion-{os.getpid()}-{time.monotonic_ns()}-"
    staged = {role: outputs[role].with_name(token + outputs[role].name) for _, role, _ in writes}
    try:
        for data, role, dtype in writes:
            _save_like_reference(
                data=data,
                reference=geometry.output_reference,
                destination=staged[role],
                dtype=dtype,
            )
        for role, path in staged.items():
            path.replace(outputs[role])
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    validation: OutputValidation = {
        "schema_version": 1,
        "lesion_voxel_count": int(lesion.sum()),
        "finite": True,
        "method": "input_grid_shape_and_finite_voxels",
    }
    return validation, compositing


def _save_like_reference(
    *,
    data: HostArray,
    reference: NiftiImage,
    destination: Path,
    dtype: type[np.generic],
) -> None:
    """Write ``data`` cropped to ``reference``'s grid, keeping its affine, qform and sform."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = nib.Nifti1Header.from_header(reference.header)
    header.set_data_dtype(dtype)
    cropped = np.asarray(data, dtype=dtype)[_input_grid(reference)]
    output = nib.Nifti1Image(cropped, reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    output.set_qform(reference.affine if qform is None else qform, int(qcode))
    output.set_sform(reference.affine if sform is None else sform, int(scode))
    nib.save(output, str(destination))


def _input_grid(reference: NiftiImage) -> tuple[slice, ...]:
    """Drop the trailing ``_pad_to_even`` voxels; the origin is unchanged."""
    return tuple(slice(0, int(size)) for size in reference.shape[:3])


def output_paths(output: Path, output_region: str = DEFAULT_OUTPUT_REGION) -> dict[str, Path]:
    """Map each output role to its file; ``box`` mode adds ``raw_synthetic_image``."""
    paths = {
        "synthetic_image": output / _SYNTHETIC_CT,
        "target_mask": output / _LESION_MASK,
        "organ_mask": output / _ORGAN_MASK,
    }
    if output_region == "box":
        paths["raw_synthetic_image"] = output / _RAW_SYNTHETIC_CT
    return paths


def localize_output(
    *,
    original: NumericArray,
    generated: NumericArray,
    bbox: tuple[slice, slice, slice],
    spacing_mm: tuple[float, float, float],
    margin_mm: float = LOCALIZATION_MARGIN_MM,
) -> tuple[FloatArray, dict[str, object]]:
    """Composite the generated CT into the original CT around the box.

    Inside ``bbox`` the generated CT is kept. A voxel whose center lies less than
    ``margin_mm`` from the box (Euclidean distance in mm, from the nearest box voxel
    center) takes ``w * generated + (1 - w) * original`` with
    ``w = 0.5 * (1 + cos(pi * d / margin_mm))``; every other voxel is the original CT
    converted to float32, which can round values from a higher-precision input.

    Args:
        original: Standardized input CT on the even-padded grid.
        generated: Full generated CT on the same grid.
        bbox: Region to keep generated, as slices; the runner passes the requested
            box widened to the lesion's bounding box.
        spacing_mm: Voxel spacing (X, Y, Z) in mm.
        margin_mm: Feather width in mm.

    Returns:
        ``(localized_ct, facts)``: a float32 array and a record of the blend policy.

    Raises:
        ModelExecutionError: If ``original`` and ``generated`` differ in shape.
    """
    if original.shape != generated.shape:
        raise ModelExecutionError("LesionDiffusion raw output shape does not match the original CT")
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    radius = np.ceil(margin_mm / spacing).astype(int)
    start = np.asarray([int(value.start) for value in bbox])
    stop = np.asarray([int(value.stop) for value in bbox])
    low = np.maximum(0, start - radius)
    high = np.minimum(np.asarray(original.shape[:3]), stop + radius)
    region = cast(
        tuple[slice, slice, slice],
        tuple(slice(int(a), int(b)) for a, b in zip(low, high, strict=True)),
    )
    squared = np.zeros(tuple(int(value) for value in high - low), dtype=np.float64)
    for axis in range(3):
        coordinates = np.arange(int(low[axis]), int(high[axis]))
        offset = (
            np.where(
                coordinates < start[axis],
                start[axis] - coordinates,
                np.where(coordinates >= stop[axis], coordinates - (stop[axis] - 1), 0),
            )
            * spacing[axis]
        )
        shape = [1, 1, 1]
        shape[axis] = offset.size
        squared = squared + (offset**2).reshape(shape)
    distance = np.sqrt(squared)
    weights = np.zeros(distance.shape, dtype=np.float64)
    affected = distance < margin_mm
    weights[affected] = 0.5 * (1.0 + np.cos(np.pi * distance[affected] / margin_mm))
    localized = np.asarray(original, dtype=np.float32).copy()
    raw = np.asarray(generated[region], dtype=np.float32)
    window = localized[region]
    transition = (weights > 0) & (weights < 1)
    window[transition] = (
        raw[transition] * weights[transition] + window[transition] * (1.0 - weights[transition])
    ).astype(np.float32)
    localized[bbox] = generated[bbox]
    return localized, {
        "policy": "bbox_generated_cosine_feather_to_original_ct",
        "margin_mm": margin_mm,
        "blend_weight": "0.5 * (1 + cos(pi * distance_mm / margin_mm))",
        "distance_metric": "euclidean_voxel_center_distance_in_physical_mm",
        "bounding_box": {
            "start_voxel": [int(value) for value in start],
            "stop_voxel_exclusive": [int(value) for value in stop],
        },
        "affected_region": {
            "start_voxel": [int(value) for value in low],
            "stop_voxel_exclusive": [int(value) for value in high],
            "weighted_voxel_count": int(np.count_nonzero(affected)),
        },
        "outside_affected_region": "original_ct_cast_float32",
    }


def _with_comma(value: str) -> str:
    # Upstream report phrases are comma-terminated.
    """Normalize an upstream report phrase to end with one comma."""
    stripped = value.strip().rstrip(",")
    return f"{stripped},"


def _json_safe(value: object) -> object:
    """Convert nested mappings, sequences, and NumPy scalars to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in cast(Sequence[object], value)]
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


__all__ = [
    "MODEL",
    "LesionDiffusionOptions",
    "load_conversion",
    "load_stage",
    "require_weights",
    "run",
    "run_native_pipeline",
    "select_lesion_component",
    "source_organ_label",
]
