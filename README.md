# thesis project "Ocean Exploration with Artificial Intelligence"

This repository contains all code used in our thesis project.


## GANs 

### GANs for Data Enhancement

We are training a `CycleGAN` to enhance our images. 

### GANs for Data Augmentation
We try different GAN versions to increase our training data:
- `DCGAN`
- `DCGAN + DiffAugment`
- `WGAN-GP + DCGAN Arch.`
- `WGAN-GP + DCGAN Arch. + DiffAugment`
- `WGAN-GP + Resnet Arch.`

#### Evaluation
We use the [offical pytorch FID implementation](https://github.com/mseitzer/pytorch-fid) to evaluate the generated images from the different GANs.

To compute the FID score between two datasets do: 
- install `pip install pytorch-fid`
- put generated and real images into two folders with paths path/to/dataset1 path/to/dataset2
- run `python -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims N`

`--dim N` changes to a different feature layer of the inception module with different dimension (choices: 64, 192, 768, 2048 default). **NOTE:** The calculation needs samples greater than the dimension of the feature layer. Our datasets are small, we use 192(?).




