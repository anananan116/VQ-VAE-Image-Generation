from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch
import os
import numpy as np
class ImageDataset(Dataset):
    def __init__(self, image_files, save_porcessed= None, transform=None):
        self.transform = transform
        if os.path.exists(f'./data_utils/raw_data/{save_porcessed}'):
            self.images = torch.load(f'./data_utils/raw_data/{save_porcessed}')
        elif save_porcessed:
            self.images = []
            for image_file in tqdm(image_files):
                self.images.append(self.load_image(image_file))
            self.images = torch.stack(self.images)
            torch.save(self.images, f'./data_utils/raw_data/{save_porcessed}')
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
    def __init__(self, top_codes, bottom_codes, for_transformer=False, hier=None):
        if for_transformer:
            if hier == 'top':
                self.top_codes = np.hstack([np.ones((top_codes.shape[0], 1), dtype=np.int64) * 512,top_codes.reshape(top_codes.shape[0],-1), np.ones((top_codes.shape[0], 1), dtype=np.int64) * 513])
                print(self.top_codes.shape), print(self.top_codes[0])
            else:
                self.top_codes = np.hstack([top_codes.reshape(top_codes.shape[0],-1)+2, np.ones((top_codes.shape[0], 1))])
            self.bottom_codes = np.hstack([bottom_codes.reshape(bottom_codes.shape[0],-1)+514, np.ones((bottom_codes.shape[0], 1))])
            self.top_codes = torch.tensor(self.top_codes, dtype=torch.long)
            self.bottom_codes = torch.tensor(self.bottom_codes, dtype=torch.long)
        else:
            self.top_codes = torch.tensor(top_codes, dtype=torch.long)
            self.bottom_codes = torch.tensor(bottom_codes, dtype=torch.long)
        self.hier = hier
    def __len__(self):
        return len(self.top_codes)
    
    def __getitem__(self, index):
        if self.hier == 'top':
            return {'input_ids': self.top_codes[index], 'labels': self.top_codes[index]}
        elif self.hier == 'bottom':
            return {'input_ids': self.top_codes[index], 'labels': self.bottom_codes[index]}
        return self.top_codes[index], self.bottom_codes[index]