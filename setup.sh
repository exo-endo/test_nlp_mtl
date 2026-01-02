#!/bin/bash
# setup.sh - Automated setup script for SDOH MTL Project

set -e  # Exit on error

echo "======================================"
echo "SDOH MTL Project Setup"
echo "======================================"
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        PYTHON_VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        echo "Found: $cmd (version $PYTHON_VERSION)"
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10+ not found. Please install Python first."
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment 'hless'..."
$PYTHON_CMD -m venv hless

echo ""
echo "Step 2: Activating virtual environment..."
source hless/bin/activate

echo ""
echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo ""
echo "Step 4: Installing dependencies from requirements.txt..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet

echo ""
echo "Step 5: Verifying installation..."
python -c "
import torch
import transformers
import streamlit
import captum
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Streamlit:', streamlit.__version__)
print('✓ Captum:', captum.__version__)
print('✓ Device: CUDA' if torch.cuda.is_available() else ('MPS (Apple Silicon)' if torch.backends.mps.is_available() else 'CPU'))
"

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future:"
echo "  source hless/bin/activate"
echo ""
echo "Quick start commands:"
echo "  python run_streamlit_mtl.py          # Launch MTL app"
echo "  python train_clinicalbert.py --epochs 2    # Train model"
echo "  python generate_wordclouds.py --max-samples 10  # Generate word clouds"
echo ""
echo "See SETUP.md for detailed documentation."
