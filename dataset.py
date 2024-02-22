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

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import os


class GetDataset(Dataset):
    def __init__(self, configuration):
        self.root_dir = configuration['root_dir']
        self.batch_size = configuration['batch_size']
        self.num_workers = configuration['num_workers']

        normalize = transforms.Normalize(
            mean=[0.5752, 0.4495, 0.4012],
            std=[0.2086, 0.1911, 0.1827]
        )

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        self.train_set = datasets.ImageFolder(
            root=os.path.join(self.root_dir, 'train'),
            transform=train_transform
        )

        self.val_set = datasets.ImageFolder(
            root=os.path.join(self.root_dir, 'test'),
            transform=val_transform
        )

        val_size = int(0.8 * len(self.val_set))
        test_size = len(self.val_set) - val_size

        self.val_set, self.test_set = random_split(self.val_set, [val_size, test_size])

    def get_data_loaders(self, shuffle_train=True):
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=shuffle_train,
            num_workers=self.num_workers, pin_memory=True
        )

        val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader


