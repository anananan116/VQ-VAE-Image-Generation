from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
import os
class CelebADataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.transform = transform
        if os.path.exists('./data_utils/raw_data/celeba_preprocessed.pt'):
            self.images = torch.load('./data_utils/raw_data/celeba_preprocessed.pt')
        else:
            self.images = []
            for image_file in tqdm(image_files):
                self.images.append(self.load_image(image_file))
            self.images = torch.stack(self.images)
            torch.save(self.images, './data_utils/raw_data/celeba_preprocessed.pt')
        
    def load_image(self, file):
        with Image.open(file) as img:
            image = self.transform(img)
            return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return the preloaded and preprocessed image
        return self.images[idx]