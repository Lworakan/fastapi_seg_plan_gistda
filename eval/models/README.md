# Place your UNet model .h5 files here

## Example structure:
```
models/
├── unet_model.h5
├── unet_epoch10.h5
├── unet_epoch20.h5
├── unet_best.h5
└── ...
```

## Model Requirements:
- Format: Keras .h5 model files
- Architecture: UNet for semantic segmentation
- Output: Binary or multi-class segmentation
- Input: RGB images (will be resized to model's expected input size)

## Naming Convention:
Use descriptive names that include:
- Model architecture: `unet_`, `deeplabv3_`, etc.
- Training details: `_epoch10`, `_best`, `_final`
- Dataset: `_deepglobe`, `_mass_`, etc.

Example: `unet_deepglobe_epoch25_best.h5`
