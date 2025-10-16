#!/bin/bash

# FastAPI Segmentation Server Startup Script
# This script sets up and runs the FastAPI server for road segmentation

echo "🚀 Starting FastAPI Road Segmentation Server"
echo "=============================================="

CONDA_DEFAULT_ENV="fastapi_seg_plan"
# Check if we're in a conda environment
echo "🐍 Using conda environment: $CONDA_DEFAULT_ENV"
    
conda activate "$CONDA_DEFAULT_ENV"


# Install dependencies
echo "📚 Installing dependencies..."
# pip install -r fastapi_requirements.txt
# pip install -r requirements.txt

# Set PYTHONPATH to current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Start the FastAPI server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📖 API documentation will be available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 fastapi_segmentation.py