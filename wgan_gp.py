##### WGAN-GP with DCGAN Generator and Discriminator and optionally with DiffAugment#####
# code inspired by https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/4.%20WGAN-GP


import argparse
import sys
sys.path.append("..")
from DiffAugment_pytorch import DiffAugment
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
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="wgan_gp", help="name of the model")
parser.add_argument("--n_epochs", type=int, default=700, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--ndf", type=int, default=64, help="size of feature maps in critic")
parser.add_argument("--ngf", type=int, default=64, help="size of feature maps in generator")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--critic_iter", type=int, default=5, help="number of critic iterations before generator update")
parser.add_argument("--lambda_gp", type=int, default=10, help="penalty coefficient")
parser.add_argument("--test", type=bool, default=False, help="if true train, if false generate images from trained model")
parser.add_argument("--num_output",type=int, default=100, help='number of generated outputs')
parser.add_argument("--diff_augment",type=bool, default=False, help='use Diffaugment or not. Default: False')
parser.add_argument("--data_real",type=bool, default=False, help='use real data or enhanced data. default: real')
parser.add_argument("--filtered_data",type=bool, default=False, help='use filtered data or all. default: all')
opt = parser.parse_args()
print(opt)



########### Models: Discriminator and Generator implementation from DCGAN paper ##########
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(opt.ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            
        )

    def forward(self, input):
        return self.main(input)

'''
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x img_size x img_size
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Conv2d below makes output into 1x1
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
'''

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.latent_dim, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( opt.ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
'''
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
            # Output: N x channels_img x img_size x img_size
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
'''

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





####################### Gradient penalty ##############################


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




######################## Training ##################################

# Decide which device we want to run on
device = "cuda" if torch.cuda.is_available() else "cpu"


# policy for diffaugment
policy = 'color,translation,cutout'

if not opt.test:
    
    # Configure data loader
    transforms = transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(opt.channels)], [0.5 for _ in range(opt.channels)]),
        ]
    )
  
    if opt.data_real == False:
        if opt.filtered_data == True:
            path = os.path.join(sys.path[0], "data_real_filtered")
        path = os.path.join(sys.path[0], "data_real")
    else:
        if opt.filtered_data == True:
            path = os.path.join(sys.path[0], "data_fake_filtered")
        path = os.path.join(sys.path[0], "data_fake")

    dataset = datasets.ImageFolder(root=path, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
    )

   


    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])

    gen = Generator().to(device)
    critic = Discriminator().to(device)

    
    initialize_weights(gen)
    initialize_weights(critic)

    print(gen)
    print(critic)
    

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    opt_critic = optim.Adam(critic.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # for saving image progress
    os.makedirs(f'output/{opt.name}', exist_ok=True) 
    fixed_noise = torch.randn(32, opt.latent_dim, 1, 1).to(device)
    
    # Lists to keep track of loss
    G_losses = []
    C_losses = []

    gen.train()
    critic.train()

    print("Starting Training Loop...")
    for epoch in range(opt.n_epochs):
        # Target labels not needed! 
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            
            #add differential augmentation if True
            if opt.diff_augment:
                real=DiffAugment(real,policy=policy)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(opt.critic_iter):
                noise = torch.randn(cur_batch_size, opt.latent_dim, 1, 1).to(device)
                fake = gen(noise)
                
                #add differential augmentation if True
                if opt.diff_augment:
                    fake=DiffAugment(fake,policy=policy)
                
                critic_real = critic(real).reshape(-1)   
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + opt.lambda_gp * gp
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

            # Save Losses for plotting later
            G_losses.append(loss_gen.item())
            C_losses.append(loss_critic.item())

            # Print losses occasionally 
            if batch_idx % 3 == 0 and batch_idx > 0:
                print(
                    f"[Epoch {epoch}/{opt.n_epochs}] [Batch {batch_idx}/{len(loader)}] \
                    [D loss: {loss_critic:.4f}], [G loss: {loss_gen:.4f}]"
                )
        #save progress images 
        with torch.no_grad():
            if epoch % 20 == 0:
                fake = gen(fixed_noise)
                save_image(fake[:25], f'output/{opt.name}/%d.png' % epoch, nrow=5, normalize=True)
        
        #save generator weights 
        if epoch % 1000 == 0:             
            state = {'generator': gen.state_dict()}
            filename=f'checkpoints/{opt.name}_{epoch}'
            torch.save(state, filename)

    with torch.no_grad():
        fake = gen(fixed_noise)
        save_image(fake[:25], f'output/{opt.name}/%d.png' % epoch, nrow=5, normalize=True)    

    #save last generator weights           
    state = {'generator': gen.state_dict()}
    filename=f'checkpoints/{opt.name}_last'
    torch.save(state, filename)

    #save losses in plot
    plt.figure(figsize=(10,5))
    plt.title(f'{opt.name}: Generator and Critic Training Loss')
    plt.plot(G_losses,label="G")
    plt.plot(C_losses,label="C")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'output/{opt.name}/loss.png')
    #plt.show()





############### Generate images from trained model #################

            
else:

    # dir to save generated images
    os.makedirs(f'result/{opt.name}', exist_ok=True)

    #load checkpoints for generator
    state_dict = torch.load(f'checkpoints/{opt.name}')
    gen = Generator().to(device)
    gen.load_state_dict(state_dict['generator'])
    
    #generate images 
    with torch.no_grad():
        for i in range(opt.num_output): 
            noise = torch.randn(1, opt.latent_dim, 1, 1).to(device) 
            fake = gen(noise)
            save_image(fake, f'result/{opt.name}/%d.png' % i, normalize=True)




               
