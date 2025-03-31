from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class GarmentDataset(Dataset):
    def __init__(self, df, minority_classes, transform_strong=None, transform_normal=None):
        self.labels = df['category_label'].values.astype(np.int64)
        self.image_paths = df['image_name'].values
        self.minority_classes = minority_classes
        self.transform_strong = transform_strong
        self.transform_normal = transform_normal

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        category_id = int(self.labels[index])
        image = Image.open("dataset/" + image_path).convert('RGB')

        if category_id in self.minority_classes:
            image = self.transform_strong(image)
        else:
            image = self.transform_normal(image)

        return image, category_id

    def __len__(self):
        return len(self.labels)