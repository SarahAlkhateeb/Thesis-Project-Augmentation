##### WGAN-GP with DCGAN Generator and Discriminator and optionally with DiffAugment#####
# code inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP

"""
Models:
Discriminator and Generator implementation from DCGAN paper
"""
import argparse
import sys
sys.path.append("..")
from GANs_Augmentation.DiffAugment_pytorch import DiffAugment
import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image



parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="wgan_gp", help="name of the model")
parser.add_argument("--n_epochs", type=int, default=700, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--critic_iter", type=int, default=5, help="number of critic iterations before generator update")
parser.add_argument("--lambda_gp", type=int, default=10, help="penalty coefficient")
parser.add_argument("--test", type=bool, default=False, help="if true train, if false generate images from trained model")
parser.add_argument("--num_output",type=int, default=100, help='number of generated outputs')
parser.add_argument("--diff_augment",type=bool, default=False, help='use Diffaugment or not. Default: False')
opt = parser.parse_args()
print(opt)




class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True), #wgan-gp papers suggests normalization schemes in critic which don’t introduce correlations between examples.  
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): 
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d,nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

"""
Gradient penalty
"""


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


"""
Training of WGAN-GP
"""
# Hyperparameters etc.

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = opt.lr
BATCH_SIZE = opt.batch_size
IMAGE_SIZE = opt.img_size
CHANNELS_IMG = opt.channels
Z_DIM = opt.latent_dim
NUM_EPOCHS = opt.n_epochs
FEATURES_CRITIC = 64 #aladin uses 16, pytorch 64
FEATURES_GEN = 32 #alading uses 16, pytorch 32
CRITIC_ITERATIONS = opt.critic_iter
LAMBDA_GP = opt.lambda_gp

# policy for diffaugment
policy = 'color,translation,cutout'

if not opt.test:
    
    # Configure data loader
    transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
        ]
    )
    #dataroot = "/home/2019/bodlak/MS-2021/datainbackup/thesis/thesis/GANs_Augmentation/data" #on server
    dataroot= "/Users/lisabodlak/Desktop/Thesis/code/GANs_Augmentation/data" #on mac
    dataset = datasets.ImageFolder(root=dataroot, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )



    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])


    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
    initialize_weights(gen)
    initialize_weights(critic)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    # for saving image progress
    os.makedirs(f'output/{opt.name}', exist_ok=True) 
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)


    gen.train()
    critic.train()

    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise)
                #add differential augmentation if True
                if opt.diff_augment:
                    real=DiffAugment(real,policy=policy)
                    fake=DiffAugment(fake,policy=policy)
                
                critic_real = critic(real).reshape(-1)   
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally 
            if batch_idx % 3 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )
        #save progress images 
        with torch.no_grad():
            if epoch % 2 == 0:
                fake = gen(fixed_noise)
                save_image(fake[:25], f'output/{opt.name}/%d.png' % epoch, nrow=5, normalize=True)

    #save generator weights           
    state = {'generator': gen.state_dict(),'params': [Z_DIM, CHANNELS_IMG, FEATURES_GEN]}
    filename=f'checkpoints/{opt.name}'
    torch.save(state, filename)


#Generate images from trained model            
else:
    # dir to save generated images
    os.makedirs(f'result/{opt.name}', exist_ok=True)

    #load checkpoints for generator
    state_dict = torch.load(f'checkpoints/{opt.name}')
    params = state_dict['params']
    gen = Generator(*params).to(device)
    gen.load_state_dict(state_dict['generator'])
    
    #generate images 
    with torch.no_grad():
        for i in range(opt.num_output): 
            noise = torch.randn(1, Z_DIM, 1, 1).to(device) 
            fake = gen(noise)
            save_image(fake, f'result/{opt.name}/%d.png' % i)




               
        
