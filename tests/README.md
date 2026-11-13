# Real CT input

The lesion-output localization tests use the converted HCC-TACE-Seg HCC_063
contrast CT at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` by default;
`MEDMLX_TEST_CT` overrides the path. The crop includes a real bone voxel above
the decoder's 1000 HU clipping limit.

Placement tests use the matching HCC_063 liver annotation at
`~/.cache/medmlx/test-data/hcc_063_liver.nii.gz`; `MEDMLX_TEST_MASK` overrides
the annotation. The CT and annotation grids are checked before use. Hollow-organ
placement branches use the liver mask as an input to test the branch arithmetic;
they do not represent a colon annotation.

The native runner tests use matched CT, liver and tumor crops. The tumor mask
defaults to `~/.cache/medmlx/test-data/hcc_063_tumor.nii.gz` and can be set with
`MEDMLX_TEST_TUMOR`. The test grid uses controlled 1 mm geometry, and the liver
extent includes tumor voxels because the source stores those labels separately.

Regenerate the independent box-placement references from the pinned upstream checkout:

```sh
PYTHONPATH=. python scripts/record_placement_reference.py --source PATH_TO_LESIONDIFFUSION
PYTHONPATH=. python scripts/record_generation_reference.py --source PATH_TO_LESIONDIFFUSION \
  --reference-adapter PATH_TO_TORCH_REFERENCE_ADAPTER
PYTHONPATH=.:src python scripts/record_encoder_reference.py --source PATH_TO_LESIONDIFFUSION \
  --reference-adapter PATH_TO_TORCH_REFERENCE_ADAPTER
```

Outputs are stored outside Git in `work/real-reference`; `MEDMLX_TEST_REFERENCES`
selects a different local reference directory.
The generation recorder uses Torch CPU and the `_upstream.py` reference adapter;
its seeded reduced networks and recorded text condition are test inputs, not
released model weights.
VQ encoder and axis-order references use CT crops, including odd and long
extents. Their recorder requires the Torch version recorded in the numerical
fixtures. Explicit latent, noise, and scalar calibration arrays remain numerical
controls.
