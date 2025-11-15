"""
The problem that the team was trying to solve was to generate images from a given image.
It was a supervised learning problem. So it couls have been solved using a Convolutional Neural Network.
But the issue was hand coding the loss function. 
So they realised if our goal is to generate images that look real, that is funcdamentally a generative problem.
And so they used a GAN to solve this problem.

The GAN that they used was inspired by U-Net style model. Conv layers decreases the size of the image and deconv layers increase the size of the image.
with skip connections to preserve the features. That is the generator.

And Discriminator is a simple CNN that classifies the image as real or fake.

It is also called as patchGAN because output comes in as 1xNx30x30 where N is the number of patches.
the input for the image being 256x256.
This is a bit different than normal GANs

Each value for the patch is a value between 0 and 1.
"""


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
