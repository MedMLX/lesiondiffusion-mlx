# lesiondiffusion-mlx

Native MLX port of LesionDiffusion for adding synthetic lesions to CT,
converted from the original LMNet, LINet and VQ checkpoints.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Model execution runs in MLX on Metal (macOS arm64). There is no PyTorch,
  MPS, NumPy or CPU fallback; host I/O and preprocessing may use CPU
  libraries.
- Unsupported checkpoints, options and inputs raise a clear error. Do not add
  compatibility shims or silent fallbacks.
- Never commit model weights, medical images, patient data or generated
  outputs.
- Keep upstream notices in THIRD_PARTY_NOTICES.md; model weights retain their
  publisher terms.
- Preserve the source sampling, crop-size and RAS 1 mm output-grid contracts.
  Synthetic output is not ground truth.

## Checks

- `make env` creates the locked development environment.
- `make qa` runs Ruff and Pyright; it does not run models.
- `make test TESTS=tests/test_name.py` runs selected tests; numerical changes
  need the affected comparison against the upstream reference.
- There is no hosted CI; local checks are authoritative.
