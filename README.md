# VQ-VAE-VQ-GAN-Image-Generation
Image generation usinng vector quantized autoencoders

Download data from celebA-HQ [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html), and put unziped images into ./data_utils/raw_data

It needs about 40-50 GB of RAM(not GRAM) to load the data and transform them. I will implement transformation on the fly so it need less ram to prepare the data in future versions.