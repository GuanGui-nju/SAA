import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

from PIL import Image
import numpy as np
import copy

import time


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

                # self.strong_transform1 = RandAugment(3, 5)
                # self.strong_transform2 = RandAugment(3, 5)
                # self.strong_transform3 = RandAugment(3, 5)
                # self.strong_transform4 = RandAugment(3, 5)

                self.strong_transform.transforms.insert(0, RandAugment(3, 5))

        else:
            self.strong_transform = strong_transform

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
            if not self.use_strong_transform:
                return img_w, target
            else:
                iters = gol.get_value('iter')

                # if iters < 100000:
                #     img_s1 = self.strong_transform1(img)
                # elif iters < 200000:
                # img_s1 = self.strong_transform1(img)
                # img_s2 = self.strong_transform2(img)
                # img_s2 = img_s2.crop((0, 0, 16, 32))
                # img_s1.paste(img_s2, (0, 0, 16, 32))
                # else:
                # img_s1 = self.strong_transform1(img)
                # img_s2 = self.strong_transform2(img)
                # img_s3 = self.strong_transform3(img)
                # img_s4 = self.strong_transform4(img)
                #
                # img_s2 = img_s2.crop((0,0,16,16))
                # img_s3 = img_s3.crop((16,0,32,16))
                # img_s4 = img_s4.crop((0,16,16,32))
                #
                # img_s1.paste(img_s2, (0, 0, 16, 16))
                # img_s1.paste(img_s3, (16, 0, 32, 16))
                # img_s1.paste(img_s4, (0, 16, 16, 32))
                #
                # img_s1 = self.strong_transform(img_s1)
                # h,w=32,32
                # img_x = img.crop((0,0,16,32))
                # t = self.strong_transform(img_x)
                # print('----')
                # print(t.shape)

                # crop_transform = transforms.RandomCrop(32, padding=4)
                # img = crop_transform(img)

                # import gol
                # iters = gol.get_value('iter')
                # print(iters)
                # from models.fixmatch.fixmatch import ITER
                # print(ITER)
                # print(read_iter())

                # print(self.flag)
                # print(time.time())
                img_s = torch.cat(
                    (self.strong_transform(img.crop((0, 0, 16, 32))), self.strong_transform(img.crop((16, 0, 32, 32)))),
                    dim=2) if \
                    torch.rand(1) > 0.5 else torch.cat(
                    (self.strong_transform(img.crop((0, 0, 32, 16))), self.strong_transform(img.crop((0, 16, 32, 32)))),
                    dim=1)

                # img_s = torch.cat((self.strong_transform(img[:, :, :, 0:int(w / 2)]), self.strong_transform(img[:, :, :, int(w / 2):w])), dim=3) if \
                #     torch.rand(1) > 0.5 else torch.cat(
                #     (self.strong_transform(img[:, :, 0:int(h / 2), :]), self.strong_transform(img[:, :, int(h / 2):h, :])), dim=2)

                # return img_w, self.strong_transform(img), target
                return img_w, img_s, target

    def __len__(self):
        return len(self.data)

    def set_iters(self):
        # print(self.flag)
        self.flag += 1
