# Release notes

## 0.2.8

### Fixes

**Conditioned inference decodes the VQ latent in FP32 to avoid FP16 overflow.** Source mode retains FP16 arithmetic and reports a clear error if decoding produces non-finite values.

### New

**Keep the input CT with `output_region="lesion"`.** This mode adds the generated-minus-source VQ reconstruction within 33 voxels of the lesion on each axis and preserves the input elsewhere, avoiding the source reconstruction shift retained by box mode.
