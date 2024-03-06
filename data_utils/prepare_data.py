from torchvision import transforms
from torch.utils.data import DataLoader, random_split
# Assuming CelebADataset is properly defined in your datasets module
from .datasets import CelebADataset
import glob

class NormalizeTransform:
    def __call__(self, x):
        return x * 2 - 1

def load_celebA(img_size, validation_ratio, test_ratio, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Adding normalization with standard ImageNet values
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    image_files = glob.glob('./data_utils/raw_data/CelebA-HQ-img/*.jpg')

    # Create the CelebA dataset with preloaded images
    dataset = CelebADataset(image_files, transform=transform)

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
