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



class Discriminator(nn.Module):
    def __init__(self, img_dim): # MNIST 28x28=784
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1), # in case of GANs, ReLU is not preferred, LeakyReLU is generally used
            nn.Linear(128, 1),
            nn.Sigmoid(), # to get output between 0 and 1
        )

    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim): # z_dim here is latent noise dim; img_dim is 784 for MNIST
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(), # to get output between -1 and 1, because we will normalize MNIST images between -1 and 1
        )

    def forward(self, x):
        return self.gen(x)
    

# Hyperparameters etc.
# GAN is incredibly sensitive to hyperparameters, so these values are important
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 3e-4 # learning rate for Adam, said by Andrej Karpathy
z_dim = 64 # latent vector dimension, we can try with 32, 64, 128 etc.
img_dim = 28 * 28 * 1 # MNIST images are 28x28 with 1 channel
batch_size = 32
num_epochs = 50
sample_interval = 500 # interval to generate and save images


disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device) # fixed noise to see the progression of generated images in tensorboard
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))] # subtract mean 0.5 and divide by std 0.5 to get range -1 to 1
)
# Actual Mean and std of MNIST are (0.1307,) and (0.3081,), but for GANs, we generally use (0.5,) and (0.5,)


# workaround for macOS/venv SSL certificate issues: set ssl context using certifi bundle
try:
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())
except Exception:
    # If certifi isn't available, torchvision will try to download and may fail with CERTIFICATE_VERIFY_FAILED.
    # If you see that error, install certifi into your venv: pip install certifi
    pass

try:
    dataset = datasets.MNIST(root="dataset/", transform = transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss eqn is -w_n* [y_n*log(x_n) + (1-y_n)*log(1-x_n)]
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
    step = 0
except Exception as e:
    # Provide a helpful error message to the user
    raise RuntimeError(
        "Failed to download MNIST. If you're on macOS with a virtual environment, try: `pip install certifi` "
        "and re-run. Original error: " + str(e)
    )


for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader): # taking real images and labels
        real = real.view(-1, 784).to(device) # flatten the images; -1 becuase we want to keep the first dimension as batch size
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device) # generate random noise; guassian distibution
        fake = gen(noise) # generate fake images from noise

        disc_real = disc(real).view(-1) # discriminator output for real images
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # real labels are 1
        # above is log(D(real)) as minimise of neg(y_n*log(x_n))
        disc_fake = disc(fake).view(-1) # discriminator output for fake images
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # fake labels are 0
        # above is log(1 - D(G(z))) as maximise of neg((1-y_n)*log(1-x_n))
        lossD = (lossD_real + lossD_fake) / 2 # total discriminator loss

        opt_disc.zero_grad()
        lossD.backward(retain_graph=True)
        # everything created in forward pass is cleared after backward(), so we need to zero_grad() again before next backward()
        # two ways to do this: fake.detach() or lossS.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output)) # we want the generator to fool the discriminator

        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # additinal code to print losses and save images in tensorboard
        if batch_idx % sample_interval == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(
                    real.reshape(-1, 1, 28, 28), normalize=True
                )

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)

            step += 1
