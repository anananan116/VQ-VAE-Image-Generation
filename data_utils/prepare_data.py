from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from .datasets import ImageDataset, latentDataset
import glob
import torch
class NormalizeTransform:
    def __call__(self, x):
        return x * 2 - 1

def load_images(img_size, validation_ratio, test_ratio, batch_size, dataset_name):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Adding normalization with standard ImageNet values
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    if dataset_name == "celebA":
        image_files = glob.glob('./data_utils/raw_data/CelebA-HQ-img/*.jpg')
    elif dataset_name == "FFHQ":
        image_files = glob.glob('./data_utils/raw_data/FFHQ-img/*.png')
    elif dataset_name == "both":
        image_files = glob.glob('./data_utils/raw_data/CelebA-HQ-img/*.jpg')
        image_files.extend(glob.glob('./data_utils/raw_data/FFHQ-img/*.png'))
    else:
        raise ValueError("Invalid dataset")

    # Create the CelebA dataset with preloaded images
    dataset = ImageDataset(image_files, transform=transform)

    dataset_size = len(dataset)
    validation_size = int(validation_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - validation_size - test_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader

def load_latent_code(validation_ratio, batch_size):
    top_codes = np.load("t_codes.npy").astype(np.int64)
    bottom_codes = np.load("b_codes.npy").astype(np.int64)
    dataset = latentDataset(top_codes, bottom_codes)
    dataset_size = len(dataset)
    validation_size = int(validation_ratio * dataset_size)
    train_size = dataset_size - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader