from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
import os
class ImageDataset(Dataset):
    def __init__(self, image_files, save_porcessed= False, transform=None):
        self.transform = transform
        if os.path.exists('./data_utils/raw_data/preprocessed.pt'):
            self.images = torch.load('./data_utils/raw_data/preprocessed.pt')
        elif save_porcessed:
            self.images = []
            for image_file in tqdm(image_files):
                self.images.append(self.load_image(image_file))
            self.images = torch.stack(self.images)
            torch.save(self.images, './data_utils/raw_data/preprocessed.pt')
        else:
            self.images = []
            for image_file in tqdm(image_files):
                self.images.append(self.load_image(image_file))
            self.images = torch.stack(self.images)

    def load_image(self, file):
        with Image.open(file) as img:
            image = self.transform(img)
            return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return the preloaded and preprocessed image
        return self.images[idx]

class latentDataset(Dataset):
    def __init__(self, top_codes, bottom_codes):
        self.top_codes = torch.tensor(top_codes, dtype=torch.long)
        self.bottom_codes = torch.tensor(bottom_codes, dtype=torch.long)
    
    def __len__(self):
        return len(self.top_codes)
    
    def __getitem__(self, index):
        return self.top_codes[index], self.bottom_codes[index]