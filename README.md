# StarNet Arrakis Finetuning

Finetuning StarNet v1 (TensorFlow 2) for removing stars from Arrakis telescope images.

## What is this?

This is a modified version of [StarNet v1](https://github.com/nekitmm/starnet) that:
- ✅ Works with **TensorFlow 2.18** and **Keras 3.0**
- ✅ Supports **.tiff** file extensions (in addition to .tif)
- ✅ Includes ready-to-use finetuning script for custom datasets
- ✅ Optimized for **Kubeflow** and **GPU training**
- ✅ Tested on **NVIDIA H100** (but works on any CUDA GPU)

## Key Modifications

### 1. Keras 3.0 Compatibility Fixes
The original StarNet code used TensorFlow operations that don't work with Keras 3.0. We fixed:
- Replaced `tf.concat()` with `L.Concatenate()` layer
- Replaced `tf.math.subtract()` with `L.Subtract()` layer
- Replaced `tf.nn.sigmoid()` with `L.Activation('sigmoid')` layer

See `starnet_v1_TF2.py` for all changes.

### 2. TIFF Extension Support
Modified `starnet_utils.py` and `starnet_v1_TF2.py` to accept both `.tif` and `.tiff` file extensions.

### 3. Ready-to-Use Finetuning Script
Created `finetune_arrakis.py` with:
- Pre-configured hyperparameters for finetuning
- CUDA library path configuration
- Clear progress output
- Automatic model saving

## Quick Start (Kubeflow)

### 1. Clone this repository in JupyterLab Terminal

```bash
git clone https://github.com/YOUR_USERNAME/starnet-arrakis-finetune.git
cd starnet-arrakis-finetune
```

### 2. Run the setup script

```bash
bash setup_kubeflow.sh
```

This will:
- Check GPU availability
- Install Miniconda (if needed)
- Create conda environment with TensorFlow + CUDA
- Download pre-trained weights (~401MB)

### 3. Upload your training data

Copy your paired images to:
- `train/original/` - Images with stars
- `train/starless/` - Images without stars (same filenames)

Supported formats: `.tif`, `.tiff` (8-bit or 16-bit)

### 4. Start training

```bash
conda activate starnet
python finetune_arrakis.py
```

## Training Configuration

Edit `finetune_arrakis.py` to adjust:

```python
MODE = 'RGB'           # or 'Greyscale'
WINDOW_SIZE = 256      # Patch size (256, 512, or 1024)
STRIDE = 128           # Stride for inference
LEARNING_RATE = 1e-4   # Learning rate for finetuning
EPOCHS = 20            # Number of training epochs
BATCH_SIZE = 1         # Batch size
```

### Recommended Settings

**For H100 or A100 (20GB+ VRAM):**
```python
WINDOW_SIZE = 512
BATCH_SIZE = 4
```

**For RTX 4090 / V100 (10-16GB VRAM):**
```python
WINDOW_SIZE = 512
BATCH_SIZE = 1
```

**For RTX 3060 / 4060 (6-8GB VRAM):**
```python
WINDOW_SIZE = 256
BATCH_SIZE = 1
```

## Expected Training Time

With **NVIDIA H100** (22GB):
- ~15-20 minutes for 20 epochs

With **NVIDIA RTX 4060** (8GB):
- ~2-3 hours for 20 epochs

## Output Files

After training completes:
- `starnet_arrakis_finetuned_G_RGB.h5` - Finetuned Generator
- `starnet_arrakis_finetuned_D_RGB.h5` - Finetuned Discriminator
- `history_arrakis_RGB.pkl` - Training history

## Using the Finetuned Model

```python
from starnet_v1_TF2 import StarNet

# Load finetuned model
model = StarNet(mode='RGB', window_size=512, stride=256)
model.load_model(weights='./starnet_arrakis_finetuned')

# Remove stars from an image
model.transform('input_image.tif', 'output_starless.tif')
```

## Troubleshooting

### Out of Memory Error
Reduce `WINDOW_SIZE` and/or `BATCH_SIZE` in `finetune_arrakis.py`.

### CUDA libdevice not found
The script automatically sets `XLA_FLAGS` to point to the conda environment. If you still see this error, manually set:
```bash
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/path/to/conda/envs/starnet'
```

### Training is very slow
Check that GPU is being used:
```bash
nvidia-smi
```
You should see Python process using GPU memory.

## Requirements

- NVIDIA GPU with CUDA support
- ~2GB disk space (for conda environment + weights)
- Training data in paired format (original/starless)

## Credits

- Original StarNet by [Nikita Misiura](https://github.com/nekitmm/starnet)
- TensorFlow 2 / Keras 3.0 compatibility fixes by Claude Code
- Arrakis telescope data finetuning setup

## License

This code inherits the license from the original StarNet repository. Please review the original LICENSE file.

## Citation

If you use this code, please cite the original StarNet paper and repository.
