import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class CropDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = ["damaged", "non_damaged"]
        self.transform = transform

        data_path = os.path.join(root, "crop")

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        # image = cv2.imread(image_path)
        # image = torch.from_numpy(image)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    dataset = CropDataset(root="./data", train=True, transform=transform)

    image, label = dataset.__getitem__(100)


    print(image.shape)


    print(label)
