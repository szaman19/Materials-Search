import argparse
import os
import pickle
import time
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gan_logger import GANLogger
from mof_dataset_v2 import MOFDatasetV2
from sphere_dataset import SphereDataset

folder = Path("mof_wgan_gp_multi_channel")
images_folder = folder / "images"
os.makedirs(images_folder, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples")

adam_b1 = 0.5
adam_b2 = 0.9  # or 0.999
clip_value = 0.01

opt = parser.parse_args()
GANLogger.log(opt)

grid_size = 32
channels = 1
img_shape = (channels, grid_size, grid_size, grid_size)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(opt.latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.shape[0], *img_shape)
#         return img


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        kernel_size = 4
        stride = 2
        padding = 1

        self.latent_to_features = nn.Sequential(
            nn.Linear(opt.latent_dim, 8 * grid_size * channels * 2 * 2 * 2),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose3d(grid_size * 8, grid_size * 4, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 4),
            nn.ReLU(),

            nn.ConvTranspose3d(grid_size * 4, grid_size * 2, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 2),
            nn.ReLU(),

            nn.ConvTranspose3d(grid_size * 2, grid_size, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size),
            nn.ReLU(),

            nn.ConvTranspose3d(grid_size, channels, kernel_size, stride, padding),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, z):
        # print("ZIN:", z.shape)
        z = self.latent_to_features(z)
        # print("ZIN2:", z.shape)
        z = z.view(z.shape[0], channels * grid_size * 8, 2, 2, 2)
        # print("ZIN3:", z.shape)
        return self.features_to_image(z)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         # padding_amount = grid_size // 10
#         # padding_size = (padding_amount * 2) ** 2
#         # self.periodic_pad = PeriodicPad3d(padding_amount)
#
#         # DON'T USE BATCH NORM WITH GP
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, x: torch.Tensor):
#         # print("BEFORE:", x.shape)
#         # x = self.periodic_pad(x)
#         # print("AFTER:", x.shape)
#         return self.model(x.view(x.shape[0], -1))  # Shape: [batch_size x 1]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        in_channels = int(np.prod(img_shape))
        kernel = 5
        stride = 2
        padding = 3
        print("In Channels:", in_channels)

        self.model = nn.Sequential(  # DON'T USE BATCH NORM WITH GP
            nn.Conv3d(channels, grid_size, kernel, stride, padding, padding_mode='circular'),
            nn.LayerNorm([grid_size, 16, 16, 16]),
            nn.LeakyReLU(0.2),
            # nn.Dropout3d(0.5),  # TODO: Where should this go

            nn.Conv3d(grid_size, grid_size * 2, kernel, stride, padding, padding_mode='circular'),
            nn.LayerNorm([grid_size * 2, 8, 8, 8]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 2, grid_size * 4, kernel, stride, padding, padding_mode='circular'),
            nn.LayerNorm([grid_size * 4, 4, 4, 4]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 4, grid_size * 8, kernel, stride, padding, padding_mode='circular'),
            nn.LayerNorm([grid_size * 8, 2, 2, 2]),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid(),

            # Flatten then linear
        )

        self.final = nn.Sequential(
            nn.Linear(grid_size * 8 * 8, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):  # Input: [batch_size x channels x grid_size x grid_size x grid_size]
        # print("DISC INPUT:", x.shape)
        x = self.model(x)
        # print("DISC OUTPUT:", x.shape)
        x = self.final(x.view(x.shape[0], -1))
        return x


# class Discriminator(nn.Module):  # HYBRID DISCRIMINATOR
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         kernel = 5
#         stride = 2
#         padding = 3
#
#         self.model = nn.Sequential(  # DON'T USE BATCH NORM WITH GP
#             nn.Conv3d(channels, grid_size, kernel, stride, padding, padding_mode='circular'),
#             nn.LayerNorm([grid_size, 16, 16, 16]),
#             nn.LeakyReLU(0.2),
#         )
#
#         self.final = nn.Sequential(
#             nn.Linear(grid_size * (16 ** 3), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Linear(256, 1),
#         )
#
#     def forward(self, x):  # Input: [batch_size x channels x grid_size x grid_size x grid_size]
#         # print("DISC INPUT:", x.shape)
#         x = self.model(x)
#         # print("DISC OUTPUT:", x.shape)
#         x = self.final(x.view(x.shape[0], -1))
#         return x


def init_weights(m):
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Loss weight for gradient penalty
lambda_gp = 10

start = time.time()


# dataset = MOFDataset("_data/Test_MOFS.p")
# tmp: Voxel_MOF = dataset.data[0]
# print(tmp.grid_metadata)
# print(type(tmp.grid_tensor))
# with open("output", 'w+') as f:
#     for i in range(32):
#         for j in range(32):
#             for k in range(32):
#                 f.write(f"{i} {j} {k} {tmp.grid_tensor[i][j][k]}\n")
# print(PropertyCalculations.get_henrys_constant(tmp.grid_tensor))
#
# print("DIE!")
# exit(0)

class DatasetMode(Enum):
    TRAIN = 1
    TEST = 2
    SPHERE = 3


dataset_mode = DatasetMode.TRAIN
# dataset_mode = DatasetMode.TEST

if dataset_mode == DatasetMode.TRAIN:
    # data_loader = MOFDataset.get_data_loader("_data/Training_MOFS.p", batch_size=opt.batch_size)
    data_loader = MOFDatasetV2.get_data_loader("_data/Training_MOFS_v2.p", batch_size=opt.batch_size)
elif dataset_mode == DatasetMode.TEST:
    # data_loader = MOFDataset.get_data_loader("_data/Test_MOFS.p", batch_size=opt.batch_size)
    data_loader = MOFDatasetV2.get_data_loader("_data/Test_MOFS_v2.p", batch_size=opt.batch_size)
elif dataset_mode == DatasetMode.SPHERE:
    data_loader = DataLoader(
        # TensorDataset(SphereDataset.generate(200)),
        TensorDataset(SphereDataset.generate_complex(200, 3)),
        batch_size=opt.batch_size,
        shuffle=True,
    )

# exit(0)

# print(data_loader[0])
print("LOAD TIME:", (time.time() - start))
# exit(0)
# for i, item in enumerate(data_loader):
#     print(type(item), item)
#     if i >= 0:
#         exit(0)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.apply(init_weights)
discriminator.apply(init_weights)

if cuda:
    generator.cuda()
    discriminator.cuda()

GANLogger.log(generator, discriminator)

# Optimizers
# glr = 0.00001
# dlr = 0.0000004
# glr = 0.002
# dlr = 0.002
glr = 0.0001
dlr = 0.0001
optimizer_G = torch.optim.Adam(generator.parameters(), lr=glr, betas=(adam_b1, adam_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=dlr, betas=(adam_b1, adam_b2))


def compute_gradient_penalty(disc, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1, 1))).float().to(device)
    # Get random interpolation between real and fake samples
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = (alpha * real_samples + ((-alpha + 1) * fake_samples)).requires_grad_(True)
    d_interpolates = disc(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------


def main():
    fake_store = None
    previous_real = None

    batches_done = 0
    epochs = 1000
    for epoch in range(epochs):
        for i, images in enumerate(data_loader):
            if dataset_mode == DatasetMode.SPHERE:
                images: torch.Tensor = images[0]  # Sphere dataset ONLY
            real_images = images.to(device).requires_grad_(True)
            # print(images.shape)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            # noise = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], opt.latent_dim))))
            noise = torch.from_numpy(np.random.normal(0, 1, (images.shape[0], opt.latent_dim))) \
                .float().requires_grad_(True).to(device)

            # noise = torch.from_numpy(np.random.normal(0, 1, (images.shape[0], opt.latent_dim))) \
            #     .float().requires_grad_(True).to(device)

            # Generate a batch of images
            fake_images = generator(noise)
            # print("FAKE IMAGE SHAPE:", fake_images.shape)
            # When we feed the same thing over and over again
            # if fake_store is None:
            #     fake_images = fake_store = generator(noise)
            # else:
            #     fake_images = fake_store.clone().detach()[:images.shape[0]]

            real_pred = discriminator(real_images)  # Real images
            fake_pred = discriminator(fake_images)  # Fake images

            garbage = torch.from_numpy(np.random.normal(0, 1, (images.shape[0], np.prod(img_shape)))) \
                .float().requires_grad_(True).to(device).view(images.shape[0], *img_shape)
            garbage_pred = discriminator(garbage)

            if previous_real is not None:
                wasserstein_distance_real_only = abs(real_pred.mean() - previous_real.mean()).item()
                print(f"REAL ONLY EMD: {wasserstein_distance_real_only}")
            previous_real = real_pred

            # TODO: Regularize the outputs -1 1 range?

            # print(f"CURR:
            #       {str(fake_validity.mean().item()).ljust(21)} - {str(real_validity.mean().item()).ljust(21)} "
            #       f"= {fake_validity.mean() - real_validity.mean()} ")
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
            # Adversarial loss
            # We want the critic to maximize the separation between fake and real
            wasserstein_distance = abs(fake_pred.mean() - real_pred.mean()).item()
            d_loss = fake_pred.mean() - real_pred.mean() + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # # Clip weights of discriminator
            # for p in discriminator.parameters():
            #     p.data.clamp_(-clip_value, clip_value)

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(noise)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_pred = discriminator(fake_images)
                g_loss = -torch.mean(fake_pred)

                g_loss.backward()
                optimizer_G.step()

                print(f"[Epoch {epoch}/{epochs}]".ljust(16)
                      + f"[Batch {i}/{len(data_loader)}] ".ljust(14)
                      + f"[-C Loss: {'{:.4f}'.format(-d_loss.item()).rjust(11)}] "
                      + f"[G Loss: {'{:.4f}'.format(g_loss.item()).rjust(11)}] "
                      + f"[Wasserstein Distance: {round(wasserstein_distance, 3)}]")

                print("GARBAGE/REAL EMD:", abs(garbage_pred.mean() - real_pred.mean()).item())
                # NOTE: Garbage EMD should theoretically be very high relative to generated/real"
                # but we're not training to maximize that, only between generated so I guess it makes sense?
                GANLogger.update(-d_loss.item(), g_loss.item())

                if batches_done % opt.sample_interval == 0:
                    # print("GENERATOR WEIGHTS:")
                    # for name, param in generator.named_parameters():
                    #     print(name, param)
                    # print("\nDISCRIMINATOR WEIGHTS:")
                    # for name, param in discriminator.named_parameters():
                    #     print(name, param)

                    with open(f"{images_folder}/{str(batches_done).zfill(5)}.p", "wb+") as f:
                        pickle.dump(fake_images.cpu(), f)

                batches_done += opt.n_critic


def test():
    print("TEST")
    tensor: torch.Tensor = data_loader.dataset[0]
    print(tensor.max())


if __name__ == '__main__':
    title = f"MOF WGAN GP - GLR: {glr}, DLR: {dlr}, S={img_shape}, BS={opt.batch_size}"
    GANLogger.init(title, folder)
    main()
    # test()

"""
For convolutions:
    Log energy values
    Scale 0-1
    Log then scale?

Make sure no zeros
"""
