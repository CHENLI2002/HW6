import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm


batch_size = 8
epochs = 100
sample_size = 1  # fixed sample size for generator

nz = 128  # latent vector size
k = 1  # number of steps to apply to the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

to_pil_image = transforms.ToPILImage()

# Load train data
train_data = datasets.MNIST(
    root='input/data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz

        self.main = nn.Sequential(
            nn.Linear(nz, 64 * 4 * 4),
            nn.ReLU(True),
            View((1, 64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 1), stride=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784

        self.main = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=(4, 4), stride=1),
            nn.Flatten(),
            nn.Linear(484, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.main(x)

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
print('##### GENERATOR #####')
print(generator)
print('######################')
print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

optim_g = optim.Adam(generator.parameters(), lr=0.000005)
optim_d = optim.Adam(discriminator.parameters(), lr=0.000005)

criterion = nn.BCELoss()  # Binary Cross Entropy loss

losses_g = []  # to store generator loss after each epoch
losses_d = []  # to store discriminator loss after each epoch
images = []  # to store images generatd by the generator


def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)


# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)


# %%
# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)


# %%
# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)


# %%
# create the noise vector - fixed to track how GAN is trained.
noise = create_noise(sample_size, nz)
# %% md
# Q. Write training loop
# %%
torch.manual_seed(7777)


def generator_loss(output, true_label):
    return criterion(output, true_label)

def discriminator_loss(output, true_label):
    return criterion(output, true_label)


for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0

    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
        sample_noise = create_noise(sample_size, nz)
        # print(sample_noise)
        # print(sample_size)
        # print(data)
        image, label = data
        real_data_label = label_real(len(label))
        fake_data = generator(sample_noise)
        fake_data_label = label_fake(sample_size)

        discrimi_fake_output = discriminator(fake_data)
        discrimi_real_output = discriminator(image)
        optim_d.zero_grad()
        # print(discrimi_fake_output)

        fake_loss = torch.log(1 - discrimi_fake_output).mean()

        real_loss = torch.log(discrimi_real_output).mean()
        dis_loss = -(fake_loss + real_loss)
        dis_loss.backward()
        # print(dis_loss)

        optim_d.step()
        optim_g.zero_grad()

        sample_noise = create_noise(sample_size, nz)
        generator_fake_data = generator(sample_noise)

        gen_result = discriminator(generator_fake_data)

        # This line is modified for sub-problem 2 temporarily
        gen_loss = -torch.log(gen_result).mean()

        # print(gen_loss)

        gen_loss.backward()
        optim_g.step()

        loss_g += gen_loss.item()
        loss_d += dis_loss.item()

    # create the final fake image for the epoch
    generated_img = generator(noise).cpu().detach()

    # make the images as grid
    generated_img = make_grid(generated_img)

    # visualize generated images

    """
    if epoch == 0 or epoch == 49 or epoch == 99:

        plt.imshow(generated_img.permute(1, 2, 0))

        plt.title(f'epoch {epoch + 1}')
        plt.axis('off')
        plt.savefig(f'outputs/{epoch + 1}-generation.png')
        plt.show()
        """

    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"outputs/gen_img{epoch + 1}.png")
    images.append(generated_img)
    epoch_loss_g = loss_g / bi  # total generator loss for the epoch
    epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    print(f"Epoch {epoch + 1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
# %%
print('DONE TRAINING')
torch.save(generator.state_dict(), 'outputs/generator.pth')
# %%
# save the generated images as GIF file
imgs = [np.array(to_pil_image(img)) for img in images]
imageio.mimsave('outputs/generator_images.gif', imgs)
# %%
# plot and save the generator and discriminator loss
plt.figure()
plt.plot(losses_g, label='Generator loss')
plt.plot(losses_d, label='Discriminator Loss')
plt.legend()
plt.savefig('outputs/-dasdas.png')