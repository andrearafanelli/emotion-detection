import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


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

    def _get_transform(self):
        if self.train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target_class = os.path.basename(os.path.dirname(img_path))
        target = self.class_to_idx[target_class]

        if self.transform:
            image = self.transform(image)

        return image, target
