import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import json

class COCOcaption(Dataset):
    
    def __init__(
        self,
        root: str,
        ann_file: str,
        transform = None,
        target_transform = None):
        super(COCOcaption, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.ann_file = os.path.expanduser(ann_file)

        
        with open(self.ann_file, 'r') as f:
            data_ann = json.load(f)
            #变为了dict

        img2anno = dict()
        for i in data_ann['images']:
            img_name = i['file_name']
            img_id = i['id']
            img2anno[img_name] = []
            count = 0 
            for j in data_ann['annotations']:
                if img_id == j['image_id']:
                    img2anno[img_name].append(self.remove_punctuation(j['caption']))
                    count += 1
                    if count == 5:  
                        break

        self.ann_dict = img2anno
        self.ids = list(sorted(self.ann_dict.keys()))

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
        target = self.ann_dict[img_id]
        target = self.remove_punctuation(target)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


    def __len__(self):
        return len(self.ids)
    
    def remove_punctuation(self,texts):
        # 获取标点符号字符集
        punctuation = string.punctuation

        # 创建一个转换表，将标点符号字符映射到空格
        translator = str.maketrans('', '', punctuation)

        # 使用 translate 方法去除标点符号
        for i, text in enumerate(texts):
            texts[i] = text.translate(translator)

        return texts
