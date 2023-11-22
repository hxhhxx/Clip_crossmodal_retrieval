import numpy as np
import os
import clip
import torch
from torchvision.datasets import CocoDetection
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class COCOcaption(Dataset):
    
    def __init__(
        self,
        root: str,
        ann_dict: dict,
        transform = None,
        target_transform = None):
        super(COCOcaption, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.ann_file = ann_dict
        self.ids = list(sorted(self.ann_file.keys()))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.ann_file[img_id]
        if self.target_transform is not None:
            target = self.target_transform.tokenize(target)
        return imgs, text


    def __len__(self):
        return len(self.ids)
    
