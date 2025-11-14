import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ssl
import certifi
from tqdm import tqdm
from model import Discriminator, Generator, initialize_weights


# Hyperparameters etc.
# WGAN is incredibly sensitive to hyperparameters, so these values are important
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 5e-5 # learning rate for RMSprop as per WGAN paper
batch_size = 64
image_size = 64
channels_img = 1 # 1 for grayscale images like MNIST
z_dim = 100 # latent vector dimension, we can try with 32, 64, 128 etc.
features_d = 64 # number of features in discriminator
features_g = 64 # number of features in generator
num_epochs = 5
n_critic = 5 # number of critic iterations per generator iteration
clip_value = 0.01 # weight clipping value


transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]), # normalize to range -1 to 1
        # so that when we change the dataset, we don't have to change the mean and std
    ]
)

try:    
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass
train_ds = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_g).to(device)
disc = Discriminator(channels_img, features_d).to(device)
initialize_weights(gen)
initialize_weights(disc)

# initialize optimizers
optim_gen = optim.RMSprop(gen.parameters(), lr=lr)
optim_disc = optim.RMSprop(disc.parameters(), lr=lr)

# for tensorboard plotting
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device) # fixed noise to see the progression of generated images in tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(tqdm(train_loader)):
        real = real.to(device)
        cur_batch_size = real.size(0)

        # Train Discriminator: max E[disc(real)] - E[disc(fake)]
        for _ in range(n_critic):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            optim_disc.zero_grad()
            loss_disc.backward()
            optim_disc.step()

            # Clip weights of discriminator
            for p in disc.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train Generator: min -E[disc(fake)] <-> max E[disc(fake)]
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)
        output = disc(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            disc.train()