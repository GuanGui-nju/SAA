import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data, sample_stl10_lb, sample_labeled_data
from .dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms
import os, sys

one_time_download = False

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean["svhn"] = [0.4380, 0.4440, 0.4730]
mean["stl10"] = [0.44087965, 0.42790789, 0.38678672]


std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std["svhn"] = [0.1751, 0.1771, 0.1744]
std["stl10"] = [0.23089217, 0.22623343, 0.22368798]


def get_transform(mean, std, ds_name, train=True):
    if train:
        if ds_name == "stl10":
            return transforms.Compose([transforms.RandomHorizontalFlip(),                                             
                                        # transforms.RandomCrop(96, padding=4),
                                        transforms.RandomCrop(96, padding=16),                                         
                                        transforms.ToTensor(),                                                        
                                        transforms.Normalize(mean, std)])
        elif ds_name == "svhn":
            return transforms.Compose([transforms.RandomCrop(32, padding=4),                                          
                                        transforms.ToTensor(),                                                        
                                        transforms.Normalize(mean, std)])
        else: # cifar
            return transforms.Compose([ transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])

    else:
        return transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean, std)])

    
class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 nfold=0, # for stl10
                 data_dir='../data'):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        
        self.name = name.lower()
        self.train = train
        self.nfold = nfold
        self.data_dir = os.path.join(data_dir, self.name)
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], self.name, train)
        
    def get_data(self, wt_unlabel=True):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name == 'stl10':
            dset = getattr(torchvision.datasets, self.name.upper())
            if self.train:
                if not wt_unlabel:
                    dset = dset(self.data_dir, split='train', folds=self.nfold, download=one_time_download)
                    data, targets = dset.data, dset.labels
                    return data, targets
                
                dset_lb = dset(self.data_dir, split='train', folds=self.nfold, download=one_time_download)
                data_lb, targets_lb = dset_lb.data, dset_lb.labels
                dset_ulb = dset(self.data_dir, split='train+unlabeled', folds=self.nfold, download=one_time_download)
                data_ulb, targets_ulb = dset_ulb.data, dset_ulb.labels                   
                return data_lb, targets_lb, data_ulb, targets_ulb                              
            else:                                                                                    
                dset = dset(self.data_dir, split='test', download=one_time_download)
                data, targets = dset.data, dset.labels
                return data, targets                                                                 
        
        elif self.name=='svhn':
            dset = getattr(torchvision.datasets, self.name.upper())
            issplit = 'train' if self.train else 'test'
            dset = dset(self.data_dir, split=issplit, download=one_time_download)
            data, targets = dset.data, dset.labels
            return data, targets

        
        elif self.name=='imagenet':
            dset = getattr(torchvision.datasets, self.name.upper())
            
        
        else: # cifar
            dset = getattr(torchvision.datasets, self.name.upper())
            dset = dset(self.data_dir, train=self.train, download=one_time_download)
            data, targets = dset.data, dset.targets
            # import ipdb
            # ipdb.set_trace()
            return data, targets
        
    
    
    def get_dset(self, use_strong_transform=False, 
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """
        
        data, targets = self.get_data(wt_unlabel=False)
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        
        return BasicDataset(data, targets, num_classes, transform, 
                            use_strong_transform, strong_transform, onehot)
    
    
    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                            use_strong_transform=True, strong_transform=None, 
                            onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        if self.name == 'stl10':
            lb_data, lb_targets, ulb_data, ulb_targets = self.get_data(wt_unlabel=True)
            if num_labels < 1000:  #
                lb_data, lb_targets = sample_stl10_lb(lb_data, lb_targets, num_labels, num_classes)
            # else:
            #     lb_data = lb_data
            #     lb_targets = lb_targets
            # lb_data, lb_targets, _ = sample_labeled_data(lb_data, lb_targets, num_labels, self.num_classes)

        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets, 
                                                                        num_labels, num_classes, 
                                                                        index, include_lb_to_ulb)
        lb_dset = BasicDataset(lb_data, lb_targets, num_classes, 
                               transform, False, None, onehot)
        
        ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes, 
                               transform, use_strong_transform, strong_transform, onehot)
        
        
        return lb_dset, ulb_dset
