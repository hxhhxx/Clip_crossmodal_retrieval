import os
import string
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

class Flickr30k(Dataset):
    """`Flickr30k Entities <https://bryanplummer.com/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(Flickr30k, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file) as fh:
            next(fh)
            for line in fh:
                img_id, num, captions = line.strip().split("|")
                self.annotations[img_id].append(captions)

        self.ids = list(sorted(self.annotations.keys()))

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
        target = self.annotations[img_id]
        
        #wanna limit the size of target here but error happened when search relevant 
        target = self.remove_punctuation(target)
        target = self.limit_length(target)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_id


    def __len__(self) -> int:
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
    
    def limit_length(self, captions, max_length=50):
        # 限制每个描述的长度
        return [' '.join(caption.split()[:max_length]) for caption in captions]
    