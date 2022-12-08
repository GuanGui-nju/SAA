from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy
import torch
from skimage import filters



class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
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
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                # self.strong_transform.transforms.insert(0, RandAugment(3, 5))
                self.strong_transform1 = RandAugment(3, 5)
                self.strong_transform2 = RandAugment(3, 5)
        else:
            self.strong_transform = strong_transform

        self.easy_mask = torch.zeros(len(self.data))
        self.history_loss = torch.zeros(len(self.data))

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
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return idx, img_w, target
            else:
                if self.alg == 'flexmatch':
                    if self.easy_mask[idx] == False:
                        img_s = self.strong_transform1(img)
                        img_s = self.strong_transform(img_s)
                    else:
                        img_s1 = self.strong_transform1(img)
                        img_s2 = self.strong_transform2(img)

                        if torch.rand(1) > 0.5:
                            img_s2 = img_s2.crop((0, 0, 16, 32))  # 16,32|48,96
                            img_s1.paste(img_s2, (0, 0, 16, 32))
                        else:
                            img_s2 = img_s2.crop((0, 0, 32, 16))
                            img_s1.paste(img_s2, (0, 0, 32, 16))

                        img_s = self.strong_transform(img_s1)
                    return idx, img_w, img_s
                elif self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)


    def set_iters(self, it):
        self.flag = it

    def set_acc(self, acc):
        self.model_acc = acc

    def set_mask(self, idx, mask, mask2):
        select = np.where(mask==1)[0]
        select_false = np.where(mask2==1)[0]
        for i in select:
            self.easy_mask[idx[i]] = True
        for i in select_false:
            self.easy_mask[idx[i]] = False

    def update_loss(self, idx, iter_loss):
        self.history_loss[idx] = 0.001*self.history_loss[idx] + 0.999*iter_loss

    def update_mask(self):

        mean,std=0,0
        np_loss = self.history_loss.numpy()
        np_loss = np.sort(np_loss)

        g1 = g2 = filters.threshold_otsu(np_loss)

        self.easy_mask[self.history_loss<g1]=True
        self.easy_mask[self.history_loss>g2]=False

        print('>> mean:{:.3f}|| std:{:.3f} || g1:{:.3f} || g2:{:.3f} || g1_pro:{:.2f} || g2_pro:{:.2f}'.format(
            mean,std,g1,g2,(self.history_loss<g1).sum().numpy()/len(self.data),(self.history_loss>g2).sum().numpy()/len(self.data)
        ))
