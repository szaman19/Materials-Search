import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from GAN import Generator, Discriminator
from mof_dataset import MOFDataset

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, num_channels, side_length):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.side_length = side_length

        self.main = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.ConvTranspose3d(1024, side_length * 8, 4, 2, 0),
            nn.BatchNorm3d(side_length * 8),
            nn.ReLU(),

            nn.ConvTranspose3d(side_length * 8, side_length * 4, 4, 2, 1),
            nn.BatchNorm3d(side_length * 4),
            nn.ReLU(),

            nn.ConvTranspose3d(side_length * 4, side_length * 2, 4, 2, 1),
            nn.BatchNorm3d(side_length * 2),
            nn.ReLU(),

            nn.ConvTranspose3d(side_length * 2, num_channels, 4, 2, 1),  #
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_channels, grid_size):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size

        # in_channels, out_channels, kernel_size, stride, padding
        self.main = nn.Sequential(
            nn.Conv3d(num_channels, grid_size, 4, 2, 1),
            nn.BatchNorm3d(grid_size),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size, grid_size * 2, 4, 2, 1),
            nn.BatchNorm3d(grid_size * 2),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 2, grid_size * 4, 4, 2, 1),
            nn.BatchNorm3d(grid_size * 4),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 4, grid_size * 8, 4, 2, 1),
            nn.BatchNorm3d(grid_size * 8),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 8, 1, 4, 2, 1),  # kernel size
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, x.size(1))


def compute_gradient_penalty(critic: Discriminator,
                             real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1, 1))).float().to(device)  # TODO: Check
    interpolates = (alpha * real_samples + ((1 + (-alpha)) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.shape[0], 1).requires_grad_().to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def main():
    # HYPERPARAMETERS
    epochs = 500
    batch_size = 32
    alpha_gp = 10
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.9
    critic_updates_per_generator_update = 1

    # MOFS
    num_atoms = 12
    grid_size = 32

    train_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Test_MOFS.p", batch_size)

    # Initialize generator and discriminator
    generator: Generator = Generator(num_atoms, grid_size)
    discriminator: Discriminator = Discriminator(num_atoms, grid_size)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    batches_done = 0
    for epoch in range(epochs):
        for batch, mof in enumerate(train_loader):

            real_images = mof.to(device)
            discriminator_optimizer.zero_grad()
            numpy_array = np.random.normal(0, 1, (real_images.shape[0], 1024))

            z = torch.from_numpy(numpy_array).float().requires_grad_().to(device)

            fake_images = generator(z)
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images)
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + alpha_gp * gradient_penalty

            d_loss.backward()
            discriminator_optimizer.step()
            generator_optimizer.zero_grad()

            if batch % critic_updates_per_generator_update == 0:
                fake_images = generator(z)
                fake_validity = discriminator(fake_images)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                generator_optimizer.step()

                if batch % 16 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, epochs, batch, len(train_loader), d_loss.item(), g_loss.item())
                    )

                # if batch == 0:
                #     print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                if batches_done % 20 == 0:
                    pass
                    # print("Generated Structure: ")
                    # torch.set_printoptions(profile="full")
                    # print(fake_images[0].shape)

            batches_done += 1


if __name__ == '__main__':
    # G = Generator(12, 64)
    # z = torch.rand(16, 1024, 1, 1, 1, requires_grad=True)
    # X = G(z)
    # print(X.shape)
    main()
    # with open('gan-processed.p', 'rb') as f:
    #     data = pickle.load(f)
    #     print(data[0].shape)
