'''
Copyright 2024 Andrea Rafanelli.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License
'''

__author__ = 'Andrea Rafanelli'

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np


class GetDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = self._load_image_paths()
        self.transform = self._get_transform()

    def _load_image_paths(self):
        image_paths = []
        for cls in self.classes:
            class_folder = os.path.join(self.root_dir, cls)
            if os.path.isdir(class_folder):
                class_images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
                image_paths.extend(class_images)
        return image_paths

    @staticmethod
    def _get_transform():

        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        target_class = os.path.basename(os.path.dirname(img_path))
        target = self.class_to_idx[target_class]

        if self.transform:
            image = self.transform(image)

        return image, target
