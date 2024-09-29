import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, directory_path, transform=None):
        self.directory_path = directory_path
        self.transform = transform

        images, labels = [], []
        counter = 0
        for root, dirs, files in os.walk(directory_path):
            temp_list = [
                os.path.join(root, file)
                for file in files
                if file.endswith((".jpg", ".png"))
            ]
            if len(temp_list) > 0:
                images.extend(temp_list)
                labels.extend([counter] * len(temp_list))
                counter += 1

        self.images = images

        num_classes = len(set(labels))
        one_hot = F.one_hot(torch.tensor(labels), num_classes=num_classes)
        self.labels = one_hot.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
