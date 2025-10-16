# Environment Setup Guide

## Quick Start with Conda

### Method 1: Create from Environment File (Recommended)

The `environment.yml` file contains the complete environment configuration exported from the working setup.

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate fastapi_seg_plan

# Verify installation
conda list
```

### Method 2: Update Existing Environment

If you already have the `fastapi_seg_plan` environment:

```bash
# Update the environment
conda env update -f environment.yml --prune

# Activate the environment
conda activate fastapi_seg_plan
```

### Method 3: Export Your Current Environment

If you've made changes and want to update the environment file:

```bash
# Export current environment
conda env export -n fastapi_seg_plan > environment.yml

# Or export without build numbers for better cross-platform compatibility
conda env export -n fastapi_seg_plan --no-builds > environment.yml
```

## Environment Details

### Python Version
- **Python 3.12.9** (cpython)

### Key Dependencies

#### Web Framework & API
- FastAPI 0.109.0
- Uvicorn 0.27.0 (with uvloop and httptools)
- Pydantic 2.5.3
- Starlette 0.35.1

#### Deep Learning
- PyTorch 2.8.0
- TorchVision 0.23.0
- TensorBoard 2.20.0

#### Scientific Computing
- NumPy 1.26.3
- SciPy 1.16.2
- Scikit-learn 1.7.2
- Scikit-image 0.25.2
- Numba 0.62.1

#### Image Processing
- OpenCV 4.9.0.80
- Pillow 10.2.0
- Matplotlib 3.10.7

#### Geospatial (GIS)
- GDAL 3.11.4
- Rasterio 1.3.9
- Shapely 2.0.2
- GeoPy 2.4.1
- GEOS 3.14.0
- PROJ 9.7.0

#### Graph Processing
- NetworkX 3.5
- SKNW 0.15

#### Utilities
- TQDM 4.67.1
- Requests 2.32.5
- PyYAML 6.0.3

## Platform Compatibility

The environment file is configured for:
- **macOS ARM64** (Apple Silicon)
- Prefix: `/Users/worakanlasudee/miniconda3/envs/fastapi_seg_plan`

For cross-platform compatibility, you may need to adjust some package versions.

## Troubleshooting

### Issue: Environment creation fails

**Solution 1**: Try creating with no builds
```bash
conda env create -f environment.yml --no-builds
```

**Solution 2**: Use mamba (faster conda alternative)
```bash
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

### Issue: Conflicts with existing packages

**Solution**: Remove and recreate the environment
```bash
conda env remove -n fastapi_seg_plan
conda env create -f environment.yml
```

### Issue: Platform-specific packages fail

**Solution**: Create a minimal environment and install remaining packages via pip
```bash
conda create -n fastapi_seg_plan python=3.12
conda activate fastapi_seg_plan
pip install -r requirements.txt
```

## Managing the Environment

### List all environments
```bash
conda env list
```

### Activate environment
```bash
conda activate fastapi_seg_plan
```

### Deactivate environment
```bash
conda deactivate
```

### Remove environment
```bash
conda env remove -n fastapi_seg_plan
```

### Clone environment
```bash
conda create --name fastapi_seg_plan_backup --clone fastapi_seg_plan
```

## Updating Dependencies

### Update all packages
```bash
conda update --all
```

### Update specific package
```bash
conda update numpy
# or
pip install --upgrade numpy
```

### Add new package
```bash
# Using conda
conda install package_name

# Using pip
pip install package_name

# Then export updated environment
conda env export -n fastapi_seg_plan > environment.yml
```

## Development vs Production

For production deployments, consider:

1. **Pin exact versions** (already done in environment.yml)
2. **Use pip-only requirements** for Docker/containers
3. **Separate dev dependencies** from production
4. **Test on target platform** before deployment

### Export for Production (pip only)
```bash
pip freeze > requirements-prod.txt
```

## Docker Alternative

If conda is not suitable for your deployment, you can use the environment.yml as a reference to create a Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Notes

- The environment includes geospatial libraries (GDAL, GEOS, PROJ) which may require system-level dependencies on Linux
- PyTorch 2.8.0 is CPU-only in this configuration. For GPU support, install PyTorch with CUDA separately
- Some packages like `railway` may not be needed in production and can be removed

## Support

For issues with environment setup:
1. Check conda version: `conda --version` (should be >= 4.10)
2. Update conda: `conda update conda`
3. Clear conda cache: `conda clean --all`
4. Check available disk space
5. Consult conda documentation: https://docs.conda.io/
