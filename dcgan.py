#Implementation of DCGAN from Radford et al. 2016
# Code inspired by pytorch tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html 

import argparse
import os
import sys
sys.path.append("..")
from DiffAugment_pytorch import DiffAugment
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="dcgan", help="name of the model")
parser.add_argument("--n_epochs", type=int, default=700, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--ngf", type=int, default=64, help="size of feature maps in generator")
parser.add_argument("--ndf", type=int, default=64, help="size of feature maps in discriminator")
parser.add_argument("--ngpu", type=int, default=1, help="number of gpus available")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--test", type=bool, default=False, help="if true train, if false generate images from trained model")
parser.add_argument("--num_output",type=int, default=100, help='number of generated outputs')
parser.add_argument("--diff_augment",type=bool, default=False, help='use Diffaugment or not. Default: False')
parser.add_argument("--data_real",type=bool, default=False, help='use real data or enhanced data. default: real')
opt = parser.parse_args()
print(opt)




########### Models: Discriminator and Generator implementation from DCGAN paper ##########

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = opt.ngpu
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

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# policy for diffaugment
policy = 'color,translation,cutout'




######################## Training ##################################

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

if not opt.test:

    # Configure data loader
    if opt.data_real == False:
        path = os.path.join(sys.path[0], "data_real")
    else:
        path = os.path.join(sys.path[0], "data_fake")
    dataset = datasets.ImageFolder(root=path, transform=transforms.Compose(
        [transforms.Resize(opt.img_size), 
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))

    dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=opt.batch_size,
    shuffle=True)

     # Loss function
    criterion = torch.nn.BCELoss()

    # Initialize generator and discriminator 
    generator = Generator(opt.ngpu).to(device)
    discriminator = Discriminator(opt.ngpu).to(device)

    if (device.type == 'cuda') and (opt.ngpu > 1):
        generator = nn.DataParallel(generator, list(range(opt.ngpu)))
        discriminator = nn.DataParallel(discriminator, list(range(opt.ngpu)))
        criterion = nn.DataParallel(criterion, list(range(opt.ngpu)))

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # for saving image progress
    os.makedirs(f'output/{opt.name}', exist_ok=True) 
    fixed_noise = torch.randn(32, opt.latent_dim, 1, 1).to(device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    
    # Lists to keep track of loss
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(opt.n_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            ##add differential augmentation if True
            if opt.diff_augment:
                real_cpu=DiffAugment(real_cpu,policy=policy)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, opt.latent_dim, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            ##add differential augmentation if True
            if opt.diff_augment:
                fake=DiffAugment(fake,policy=policy)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

             # Print training stats occasionally 
            if i % 3 == 0 and i > 0:
                print(
                "[Epoch %d/%d] [Batch %d/%d] \
                [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), errD.item(), errG.item()))

        #save progress images on fixed noise every second epoch
        with torch.no_grad():
            if epoch % 20 == 0:
                fake = generator(fixed_noise)
                save_image(fake[:25], f'output/{opt.name}/%d.png' % epoch, nrow=5, normalize=True)

        #save generator weights 
        if epoch % 1000 == 0:             
            state = {'generator': generator.state_dict()}
            filename=f'checkpoints/{opt.name}_{epoch}'
            torch.save(state, filename)

    with torch.no_grad():
        fake = generator(fixed_noise)
        save_image(fake[:25], f'output/{opt.name}/%d.png' % epoch, nrow=5, normalize=True)        

    #save generator weights           
    state = {'generator': generator.state_dict()}
    filename=f'checkpoints/{opt.name}_last'
    torch.save(state, filename)

    #save losses in plot
    plt.figure(figsize=(10,5))
    plt.title(f'{opt.name}: Generator and Discriminator Training Loss')
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'output/{opt.name}/loss.png')
    #plt.show()


######################## Generate images from trained model  ###################################
else:
    # dir to save generated images
    os.makedirs(f'result/{opt.name}', exist_ok=True)


    #load checkpoints for generator
    state_dict = torch.load(f'checkpoints/{opt.name}')
    
    generator = Generator(opt.ngpu).to(device)
    if (device.type == 'cuda') and (opt.ngpu > 1):
        generator = nn.DataParallel(generator, list(range(opt.ngpu)))
        
    generator.load_state_dict(state_dict['generator'])

    
    #generate images 
    with torch.no_grad():
        for i in range(opt.num_output): 
            noise = torch.randn(1, opt.latent_dim, 1, 1).to(device) 
            fake = generator(noise)
            save_image(fake, f'result/{opt.name}/%d.png' % i)