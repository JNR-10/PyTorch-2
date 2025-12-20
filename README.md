# Deep Learning Implementations in PyTorch

This repository contains implementations of various deep learning models from scratch using PyTorch. The models cover different domains including Computer Vision (CNNs), Generative Adversarial Networks (GANs), and Sequence-to-Sequence (Seq2Seq) models.

## Structure

The codebase is organized into the following directories within the `Code` folder:

### 1. CNNs (Convolutional Neural Networks)
Located in `Code/CNNs/`, this directory contains implementations of popular CNN architectures and their training scripts:
- **EfficientNet**: `EfficientNet.py`
- **InceptionNet**: `InceptionNet.py`
- **LeNet**: `LeNet.py` (Classic architecture for MNIST)
- **ResNet**: `ResNet.py`
- **VGG**: `VGG.py`

Training scripts are provided for each model (e.g., `train_efficientnet.py`, `train_resnet.py`).

### 2. GANs (Generative Adversarial Networks)
Located in `Code/GANs/`, this directory features various GAN implementations:
- **DCGAN**: Deep Convolutional GAN
- **ESRGAN**: Enhanced Super-Resolution GAN
- **SRGAN**: Super-Resolution GAN
- **SimpleGAN**: Basic GAN implementation
- **WGAN**: Wasserstein GAN
- **WGAN-GP**: Wasserstein GAN with Gradient Penalty

Each subdirectory typically contains model definitions and training logic.

### 3. Seq2Seq (Sequence to Sequence)
Located in `Code/Seq2Seq/`, this directory includes models for sequence tasks:
- **Seq2Seq_simple**: Basic Encoder-Decoder architecture
- **Seq2Seq_attention**: Encoder-Decoder with Attention mechanism
- **Seq2Seq_transformer**: Transformer-based models
- **Transformer**: A standalone implementation in `transformer_from_scratch.py`

## Requirements

Dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

## Usage

Navigate to the specific directory of the model you want to run and execute the training script. For example:

```bash
cd Code/CNNs
python train_lenet_mnist.py
```