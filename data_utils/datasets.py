from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import os
import pickle
class CelebADataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.transform = transform
        # Preload images
        if (os.path.exists('data_utils/raw_data/celeba_preprocessed.pkl')):
            with open('data_utils/raw_data/celeba_preprocessed.pkl', 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.images = []
            for image_file in tqdm(image_files):
                self.images.append(self.load_image(image_file))
            with open('data_utils/raw_data/celeba_preprocessed.pkl', 'wb') as f:
                pickle.dump(self.images, f)
        
    def load_image(self, file):
        with Image.open(file) as img:
            image = self.transform(img)
            return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return the preloaded and preprocessed image
        return self.images[idx]