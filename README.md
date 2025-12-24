# Deep Learning Implementations in PyTorch

This repository contains implementations of various deep learning models from scratch using PyTorch. The models cover different domains including Computer Vision, Generative Adversarial Networks (GANs), Sequence-to-Sequence (Seq2Seq) models, Image Segmentation, and Variational Autoencoders (VAEs).

## Structure

The codebase is organized into the following directories within the `Code` folder:

### 1. CNNs (Convolutional Neural Networks)
Located in `Code/CNNs/`, this directory contains implementations of popular CNN architectures:
- **EfficientNet**: `EfficientNet.py` - Implementation of the EfficientNet architecture.
- **InceptionNet**: `InceptionNet.py` - Implementation of the Inception network.
- **LeNet**: `LeNet.py` - Classic LeNet-5 architecture for MNIST.
- **ResNet**: `ResNet.py` - Residual Networks implementation.
- **VGG**: `VGG.py` - VGG network implementation.

Training scripts for these models are also provided (e.g., `train_efficientnet.py`, `train_resnet.py`).

### 2. GANs (Generative Adversarial Networks)
Located in `Code/GANs/`, this directory features various GAN implementations:
- **DCGAN**: Deep Convolutional GAN.
- **ESRGAN**: Enhanced Super-Resolution GAN.
- **SRGAN**: Super-Resolution GAN.
- **SimpleGAN**: A basic Introduction to GANs.
- **WGAN**: Wasserstein GAN.
- **WGAN-GP**: Wasserstein GAN with Gradient Penalty.

### 3. Seq2Seq (Sequence to Sequence)
Located in `Code/Seq2Seq/`, this directory includes models for sequence tasks:
- **Seq2Seq_simple**: Basic Encoder-Decoder architecture using RNNs/LSTMs.
- **Seq2Seq_attention**: Encoder-Decoder with Attention mechanism.
- **Seq2Seq_transformer**: Transformer-based sequence models.
- **Transformer**: A standalone implementation in `transformer_from_scratch.py` and potentially in `Seq2Seq_transformer`.

### 4. Image Segmentation
Located in `Code/ImageSegmentation/`, this directory focuses on semantic segmentation:
- **U-Net**: A fully implementation of the U-Net architecture for biomedical image segmentation (and general segmentation tasks).
  - `model.py`: Defines the UNET architecture.
  - `train.py`: Training loop for the model.
  - `dataset.py`: dataset loading utilities.
  - `utils.py`: Helper functions.

### 5. VAE (Variational Autoencoders)
Located in `Code/VAE/`, this directory contains generative models using VAEs:
- **Standard VAE**:
  - `model.py`: Implementation of a standard Variational Autoencoder.
  - `train.py`: Training script for the VAE.
- **Lightning VAE**:
  - `lightning_vae/`: Implementation using PyTorch Lightning for more structured training.

## Requirements

Dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

## Usage

Navigate to the specific directory of the model you want to run and execute the training script.

**Example: Training a CNN**
```bash
cd Code/CNNs
python train_lenet_mnist.py
```

**Example: Training U-Net**
```bash
cd Code/ImageSegmentation
python train.py
```

**Example: Training VAE**
```bash
cd Code/VAE
python train.py
```