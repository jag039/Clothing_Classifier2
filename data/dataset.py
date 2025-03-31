from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class GarmentDataset(Dataset):
    def __init__(self, df, transform=None):
        # Load image paths and categories
        self.labels = df['category_label'].values.astype(np.int64)
        self.image_paths = df['image_name'].values
        self.n_samples = len(self.image_paths)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        category_id = int(self.labels[index])
        image = Image.open("dataset/" + image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, category_id

    def __len__(self):
        return self.n_samples