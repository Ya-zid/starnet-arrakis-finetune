# Kubeflow Quick Start Guide

Follow these steps to run StarNet finetuning on Kubeflow with your H100 GPU.

## Step 1: Access JupyterLab

1. Open your Kubeflow notebook server
2. Click "Connect" to access JupyterLab
3. Open a Terminal (File → New → Terminal)

## Step 2: Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/starnet-arrakis-finetune.git
cd starnet-arrakis-finetune
```

## Step 3: Run Setup

```bash
bash setup_kubeflow.sh
```

This will take ~5-10 minutes to:
- Check GPU
- Install Miniconda
- Create conda environment
- Download pre-trained weights

## Step 4: Upload Training Data

You have two options:

### Option A: Upload via JupyterLab UI
1. In JupyterLab file browser, navigate to `starnet-arrakis-finetune/train/original/`
2. Click Upload button and select your original images (with stars)
3. Navigate to `starnet-arrakis-finetune/train/starless/`
4. Upload your starless images (must have same filenames!)

### Option B: Copy from existing location
If your data is already on the server:
```bash
cp /path/to/your/arrakihs/original/*.tiff train/original/
cp /path/to/your/arrakihs/starless/*.tiff train/starless/
```

## Step 5: Verify Data

```bash
ls train/original/ | wc -l
ls train/starless/ | wc -l
```

Both should show the same number (e.g., 32 files).

## Step 6: Start Training

```bash
conda activate starnet
python finetune_arrakis.py
```

## Expected Output

You should see:
```
============================================================
StarNet Finetuning on Arrakis Data
============================================================

1. Initializing StarNet model...

2. Loading Arrakis training dataset...
Total training images found: 32
Total size of training images: 134.22 MP
One epoch is set to 2048 iterations
Training dataset has been successfully loaded!

3. Loading pre-trained weights...
   Pre-trained weights loaded successfully!

4. Starting finetuning for 20 epochs...
   Learning rate: 0.0001
   Window size: 256x256
   Batch size: 1

============================================================
Training started... (This will take a while)
============================================================

Epoch: 0. Iteration 0 / 2048
Epoch: 0. Iteration 1 / 2048 Loss 13.671736
...
```

## Training Time on H100

With 32 training images and 20 epochs:
- **~15-20 minutes total**

Much faster than on a laptop!

## Monitoring GPU Usage

Open a new terminal and run:
```bash
watch -n 1 nvidia-smi
```

You should see:
- GPU memory usage increasing
- GPU utilization at 90-100%
- Python process using the GPU

## After Training Completes

Your finetuned models will be saved as:
```
starnet_arrakis_finetuned_G_RGB.h5
starnet_arrakis_finetuned_D_RGB.h5
history_arrakis_RGB.pkl
```

## Download Results

To download the trained models:
1. In JupyterLab file browser, right-click on the .h5 files
2. Select "Download"
3. Save to your local machine

## Test Your Model

Create a new file `test_model.py`:

```python
from starnet_v1_TF2 import StarNet

# Load finetuned model
model = StarNet(mode='RGB', window_size=512, stride=256)
model.load_model(weights='./starnet_arrakis_finetuned')

# Test on an image
model.transform('test_input.tif', 'test_output_starless.tif')
```

Run it:
```bash
conda activate starnet
python test_model.py
```

## Troubleshooting

### "conda: command not found"
Close and reopen your terminal, or run:
```bash
source ~/.bashrc
```

### "No space left on device"
Kubeflow notebooks have limited storage. Clean up:
```bash
# Remove conda package cache
conda clean --all -y

# Remove downloaded weights archive if it exists
rm -f starnet_weights2.zip
```

### Training is slower than expected
Check GPU is being used:
```bash
nvidia-smi
```

If no GPU usage, check CUDA is available:
```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Tips

1. **Leave browser open**: Keep the JupyterLab tab open during training
2. **Monitor progress**: Check terminal output every few minutes
3. **Save frequently**: Backups are auto-saved every 2 epochs to `starnet_backup_*.h5`
4. **Experiment**: Try different WINDOW_SIZE values (256, 512) to see what fits in memory

## Need More Epochs?

If results aren't good enough after 20 epochs, continue training:

1. Edit `finetune_arrakis.py` to load your finetuned model:
   ```python
   model.load_model(weights='./starnet_arrakis_finetuned', history='./history_arrakis')
   ```

2. Set `EPOCHS = 30` (or more)

3. Run again:
   ```bash
   python finetune_arrakis.py
   ```

The model will continue from where it left off!
