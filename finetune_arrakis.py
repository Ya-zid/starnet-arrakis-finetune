#!/usr/bin/env python
"""
Finetune StarNet on Arrakis images
This script will load pre-trained weights and continue training on the Arrakis dataset
"""

import os
# Set XLA CUDA path to conda environment
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/yazid/miniconda3/envs/starnet'

from starnet_v1_TF2 import StarNet

# Configuration
MODE = 'RGB'  # or 'Greyscale' depending on your data
WINDOW_SIZE = 256  # Size of training patches (reduced to fit in GPU memory)
STRIDE = 128  # Stride for tiling during inference
LEARNING_RATE = 1e-4  # Lower learning rate for finetuning
TRAIN_FOLDER = './train/'
EPOCHS = 20  # Number of epochs to finetune (adjust as needed)
BATCH_SIZE = 1  # Batch size (increase if you have enough GPU memory)

print("=" * 60)
print("StarNet Finetuning on Arrakis Data")
print("=" * 60)

# Initialize StarNet
print("\n1. Initializing StarNet model...")
model = StarNet(
    mode=MODE,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    lr=LEARNING_RATE,
    train_folder=TRAIN_FOLDER,
    batch_size=BATCH_SIZE
)

# Load training dataset
print("\n2. Loading Arrakis training dataset...")
model.load_training_dataset()

# Load pre-trained weights
print("\n3. Loading pre-trained weights...")
try:
    model.load_model(weights='./weights', history='./history')
    print("   Pre-trained weights loaded successfully!")
except Exception as e:
    print(f"   Warning: Could not load pre-trained weights: {e}")
    print("   Initializing model from scratch...")
    model.initialize_model()

# Start finetuning
print(f"\n4. Starting finetuning for {EPOCHS} epochs...")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Window size: {WINDOW_SIZE}x{WINDOW_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print("\n" + "=" * 60)
print("Training started... (This will take a while)")
print("=" * 60 + "\n")

model.train(
    epochs=EPOCHS,
    augmentation=True,  # Use data augmentation
    plot_progress=False,  # Set to True if you want live plots (requires display)
    plot_interval=50,
    save_backups=True,  # Save backups every 2 epochs
    warm_up=False  # Set to True for identity mapping warm-up
)

# Save finetuned model
print("\n5. Saving finetuned model...")
model.save_model('./starnet_arrakis_finetuned', './history_arrakis')

print("\n" + "=" * 60)
print("Finetuning Complete!")
print("=" * 60)
print("\nFinetuned model saved as:")
print("  - starnet_arrakis_finetuned_G_RGB.h5 (Generator)")
print("  - starnet_arrakis_finetuned_D_RGB.h5 (Discriminator)")
print("  - history_arrakis_RGB.pkl (Training history)")
print("\nYou can now use this model to remove stars from images!")
print("\nTo test on an image, use:")
print("  model.transform('input.tif', 'output_starless.tif')")
