#!/bin/bash
# Setup script for Kubeflow environment

echo "=========================================="
echo "StarNet Arrakis Finetuning - Kubeflow Setup"
echo "=========================================="

# Check GPU
echo -e "\n1. Checking GPU availability..."
nvidia-smi

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo -e "\n2. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
else
    echo -e "\n2. Miniconda already installed"
fi

# Create conda environment
echo -e "\n3. Creating conda environment..."
conda env create -f environment-lnx-cuda.yml -y || conda env update -f environment-lnx-cuda.yml --prune -y

# Download pre-trained weights
echo -e "\n4. Downloading pre-trained weights..."
if [ ! -f "weights_G_RGB.h5" ]; then
    wget -O starnet_weights2.zip "https://www.dropbox.com/s/lcgn5gvnxpo27s5/starnet_weights2.zip?dl=1"
    unzip starnet_weights2.zip
    rm starnet_weights2.zip
    mv weights/* .
    rmdir weights
    echo "   Weights downloaded successfully!"
else
    echo "   Weights already present"
fi

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Copy your Arrakis training data to:"
echo "   - train/original/ (images with stars)"
echo "   - train/starless/ (images without stars)"
echo ""
echo "2. Activate the environment:"
echo "   conda activate starnet"
echo ""
echo "3. Run training:"
echo "   python finetune_arrakis.py"
echo "=========================================="
