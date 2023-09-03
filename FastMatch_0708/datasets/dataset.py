import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

from PIL import Image
import numpy as np
import copy
import time
from skimage import filters


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot

        self.flag = 0

        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform1 = RandAugment(3, 5)
                self.strong_transform2 = RandAugment(3, 5)
                # self.strong_transform.transforms.insert(0, RandAugment(3,5))

        else:
            self.strong_transform = strong_transform

        self.easy_mask = torch.zeros(len(self.data))
        self.history_loss = torch.zeros(len(self.data))
        self.confidence = torch.zeros(len(self.data))
        self.model_acc = 0

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                img = Image.fromarray(img)
            img_w = self.transform(img)
            size = img.size[0]
            if not self.use_strong_transform:
                return img_w, target
            else:
                if self.easy_mask[idx] == False:
                    img_s = self.strong_transform1(img)
                    img_s = self.strong_transform(img_s)
                else:
                    img_s1 = self.strong_transform1(img)
                    img_s2 = self.strong_transform2(img)

                    if torch.rand(1)>0.5:
                        img_s2 = img_s2.crop((0, 0, int(size/2), size))  #16,32|48,96
                        img_s1.paste(img_s2, (0, 0, int(size/2), size))
                    else:
                        img_s2 = img_s2.crop((0, 0, size, int(size/2)))
                        img_s1.paste(img_s2, (0, 0, size, int(size/2)))

                    img_s = self.strong_transform(img_s1)
                return img_w, img_s, target, idx

    def __len__(self):
        return len(self.data)

    def update_loss(self, idx, iter_loss):
        self.history_loss[idx] = 0.000*self.history_loss[idx] + 1*iter_loss

    def update_mask(self):
        np_loss = self.history_loss.numpy()
        np_loss = np.sort(np_loss)

        g1 = g2 = filters.threshold_otsu(np_loss)

        self.easy_mask[self.history_loss<g1]=True
        self.easy_mask[self.history_loss>g2]=False
