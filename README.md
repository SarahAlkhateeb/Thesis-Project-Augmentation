# Thesis project "Ocean Exploration with Artificial Intelligence" 

This repository contains the code used in our thesis project for Data Augmentation.


We try different GAN versions to increase our training data:
- `DCGAN`
- `DCGAN + DiffAugment`
- `WGAN-GP + DCGAN Arch.`
- `WGAN-GP + DCGAN Arch. + DiffAugment`
- TODO: `WGAN-GP + Resnet Arch.`

#### Training 
For training: 
- DCGAN: `python dcgan.py --n_epochs ? --name 'dcgan' --diff_augment False`
- DCGAN + DiffAugment: `python dcgan.py --n_epochs ? --name 'dcgan+diff' --diff_augment True`
- WGAN-GP + DCGAN Arch.: `python wgan_gp.py --n_epochs ? --name 'wgan_gp' --diff_augment False `
- WGAN-GP + DCGAN Arch. + DiffAugment: `python wgan_gp.py --n_epochs ? --name 'wgan_gp+diff' --diff_augment True`

#### Image Generation

After training model `--name`,  the checkpoint can be used to generate a batch of new images of size `--num_output`
(**Note**: model paramters have to be the same as for the trained model, since the checkpoint only loads weights!): 

- DCGAN: `python dcgan.py --name 'dcgan' --test True --num_output ? `
- DCGAN + DiffAugment: `python dcgan.py --name 'dcgan+diff' --test True --num_output ? `
- WGAN-GP + DCGAN Arch.: `python wgan_gp.py  --name 'wgan_gp' --test True --num_output ? `
- WGAN-GP + DCGAN Arch. + DiffAugment: `python wgan_gp.py --n_epochs ? --name 'wgan_gp_diff'  --name 'wgan_gp' --test True --num_output ? `

#### Evaluation
We use the [offical pytorch FID implementation](https://github.com/mseitzer/pytorch-fid) to evaluate the generated images from the different GANs.

To compute the FID score between two datasets do: 
- install `pip install pytorch-fid`
- put generated and real images into two folders with paths path/to/dataset1 path/to/dataset2
- run `python -m pytorch_fid path/to/dataset1 path/to/dataset2 --dims N`

`--dim N` changes to a different feature layer of the inception module with different dimension (choices: 64, 192, 768, 2048 default). **NOTE:** The calculation needs samples greater than the dimension of the feature layer. Our datasets are small, we use 192(?).




