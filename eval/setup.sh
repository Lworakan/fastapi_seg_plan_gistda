#!/bin/bash

# Setup script for UNet evaluation environment

echo "=================================================="
echo "🛠️  UNet Evaluation Setup"
echo "=================================================="

# Check Python version
echo "📍 Checking Python version..."
python --version

# Create virtual environment (optional)
if [ ! -d "venv_eval" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv venv_eval
    echo "✅ Virtual environment created: venv_eval"
    echo "   Activate with: source venv_eval/bin/activate"
else
    echo "✅ Virtual environment already exists"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📝 Next steps:"
echo "   1. Place your .h5 model in models/"
echo "   2. Place test images in test_images/"
echo "   3. Run: python quick_test.py"
echo ""
echo "📚 For full documentation, see: README.md"
echo "=================================================="
