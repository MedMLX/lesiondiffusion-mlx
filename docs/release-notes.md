# Release notes

## Unreleased

**`vq_mode="slab"` runs large CTs with lower MLX memory.** In conditioned
inference with the reference decoder, it stages VQ activations on temporary
disk storage and retains whole-volume normalization and reconstruction.
A 440×440×248 abdominal CT used 2.81 GB peak MLX memory on M1 Max with DDIM20.

## 0.3.0

### Breaking changes

**`run` returns the shared `medmlx-core` result with a new `artifacts` entry, and the output roles are renamed.** `outputs` keeps the same files under `image` (was `synthetic_image`), `mask` (was `target_mask`) and `raw_image` (was `raw_synthetic_image`); `organ_mask` is unchanged. `artifacts` describes each output, and `details` keeps the run metadata.

```diff
-from lesiondiffusion_mlx.medmlx import run
+from lesiondiffusion_mlx import run
 result = run(image_path="ct.nii.gz", organ_mask_path="liver.nii.gz", output_dir="out", organ="liver", lesion_type="tumor")
-print(result["outputs"]["synthetic_image"], result["outputs"]["target_mask"])
+print(result["outputs"]["image"], result["outputs"]["mask"])
```

**Requires `medmlx-core>=0.3,<0.4`** (was 0.2.x).

### Fixes

**Outputs are restored if publishing the new files fails part-way.**

### New

**`load_model()` loads the weights once for repeated draws** on one CT and lesion outline, for example many seeds or phenotype texts. Each `model.run(...)` takes the same arguments as `run` and returns the same result; the validated geometry and encoded CT latent are reused.

**`sampler="dpmpp_2m"` selects deterministic DPM-Solver++ 2M** and `lmnet_diffusion_steps` / `linet_diffusion_steps` set the step count of each stage. The default 200-step DDIM path is unchanged, and source mode keeps DDIM.

**`decoder_mode="scaled_fp16"` decodes in scaled half precision** on M5-class GPUs with NAX enabled. Lesion-region HU differs from the default FP32 decoder by about 0.12 HU on average (maximum below 0.5 HU).

**The README example now uses `output_region="lesion"`,** which keeps the input CT outside the lesion neighborhood. The default stays `"full"`.

### Faster

**With `sampler="dpmpp_2m"` at 10 steps, a generated-mask run takes 3.5 s warm at 2.60 GB peak MLX memory with `decoder_mode="scaled_fp16"`,** against 4.7 s at 5.19 GB with the default FP32 decoder (M5 Max, 160x160x128 crop). The VQ encoder, LINet and VQ decoder now load one after another, which lowers the scaled path's peak.

## 0.2.8

### Fixes

**Conditioned inference decodes the VQ latent in FP32 to avoid FP16 overflow.** Source mode retains FP16 arithmetic and reports a clear error if decoding produces non-finite values.

### New

**Keep the input CT with `output_region="lesion"`.** This mode adds the generated-minus-source VQ reconstruction within 33 voxels of the lesion on each axis and preserves the input elsewhere, avoiding the source reconstruction shift retained by box mode.
