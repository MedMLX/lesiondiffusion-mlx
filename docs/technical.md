# Technical notes

## Peak memory

The README's peak MLX memory figures for `lesiondiffusion` are **5.19 GB** in the default
conditioned mode and **3.46 GB** in source mode, on a 160×160×128, 1 mm RAS crop.

Both measurements used an AMOS case 0008 crop and 200 diffusion steps: FP32 LMNet and
text encoders, FP16 LINet and VQ encoder, and an FP32 VQ decoder in conditioned mode
(FP16 in source mode). `output_region="lesion"` decodes twice in sequence without
raising the peak. They do not measure larger volumes. Peak MLX memory excludes host
preprocessing and other applications.
