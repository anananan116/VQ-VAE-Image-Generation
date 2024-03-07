from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
class CelebADataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.transform = transform
        self.images = []
        for image_file in tqdm(image_files):
            self.images.append(self.load_image(image_file))
        
    def load_image(self, file):
        with Image.open(file) as img:
            image = self.transform(img)
            return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return the preloaded and preprocessed image
        return self.images[idx]