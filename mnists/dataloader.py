import numpy as np
from PIL import Image, ImageColor
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import tensor
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import glob
import os
import pandas as pd


class ColoredMNIST(Dataset):
    def __init__(self, train, color_var=0.02):
        # get the colored mnist
        self.data_path = 'mnists/data/colored_mnist/mnist_10color_jitter_var_%.03f.npy'%color_var
        data_dic = np.load(self.data_path, encoding='latin1', allow_pickle=True).item()

        if train:
            self.ims = data_dic['train_image']
            self.labels = tensor(data_dic['train_label'], dtype=torch.long)
        else:
            self.ims = data_dic['test_image']
            self.labels = tensor(data_dic['test_label'], dtype=torch.long)

        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32), Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ])

    def __getitem__(self, idx):
        ims, labels = self.T(self.ims[idx]), self.labels[idx]

        ret = {
            'ims': ims,
            'labels': labels,
        }
        print("ColoredMNIST", ret)

        return ret

    def __len__(self):
        return self.ims.shape[0]

class DoubleColoredMNIST(Dataset):

    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels
        print("DoubleColoredMNIST labels:", self.labels)

        # colors generated by https://mokole.com/palette.html
        colors1 = [
            'darkgreen', 'darkblue', '#b03060',
            'orangered', 'yellow', 'burlywood', 'lime',
            'aqua', 'fuchsia', '#6495ed',
        ]
        # shift colors by X
        colors2 = [colors1[i-6] for i in range(len(colors1))]

        def get_rgb(x):
            t = torch.tensor(ImageColor.getcolor(x, "RGB"))/255.
            return t.view(-1, 1, 1)

        self.background_colors = list(map(get_rgb, colors1))
        self.object_colors = list(map(get_rgb, colors2))

        self.T = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_color = self.background_colors[i]
        back_color += torch.normal(0, 0.01, (3, 1, 1))

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_color = self.object_colors[i]
        obj_color += torch.normal(0, 0.01, (3, 1, 1))

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None,:], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_color) + (1 - im_digit)*back_color

        ret = {
            'ims': self.T(ims),
            'labels': self.labels[idx],
        }
        print("DoubleColoredMNIST", ret)
        return ret

    def __len__(self):
        return self.labels.shape[0]

class WildlifeMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150

        # get mnist
        mnist = datasets.MNIST('mnists/data', train=True, download=True)
        if train:
            ims, labels = mnist.data[:50000], mnist.targets[:50000]
        else:
            ims, labels = mnist.data[50000:], mnist.targets[50000:]

        self.ims_digit = torch.stack([ims, ims, ims], dim=1)
        self.labels = labels
        print("WildlifeMNIST labels:", self.labels)

        # texture paths
        background_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'background'
        self.background_textures = sorted([im for im in background_dir.glob('*.jpg')])
        object_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'object'
        self.object_textures = sorted([im for im in object_dir.glob('*.jpg')])

        self.T_texture = transforms.Compose([
            transforms.Resize((inter_sz, inter_sz), Image.NEAREST),
            transforms.RandomCrop(self.mnist_sz, padding=3, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        # get textures
        i = self.labels[idx] if self.train else np.random.randint(10)
        back_text = Image.open(self.background_textures[i])
        back_text = self.T_texture(back_text)

        i = self.labels[idx] if self.train else np.random.randint(10)
        obj_text = Image.open(self.object_textures[i])
        obj_text = self.T_texture(obj_text)

        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)
        im_digit = F.interpolate(im_digit[None, :], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_text) + (1 - im_digit)*back_text

        ret = {
            'ims': ims,
            'labels': self.labels[idx],
        }
        print("WildlifeMNIST", ret)
        return ret

    def __len__(self):
        return self.labels.shape[0]
    
    
#=========EURECOM DATASET CLASS========================    

"""
dl_train = DataLoader(Eurecom(root = "/home/local/AD/cordun1/experiments/data/Eurecom_Thermal", 
            annots_csv = "/home/local/AD/cordun1/experiments/data/labels/Eur_labels2_cfg.csv", 
            train=True),
            batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
"""
    
class Eurecom(Dataset):
    # Adopted for my Eurecom dataset
    
    def __init__(self, annots_csv, root, train=True):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150
        self.annots = pd.read_csv(annots_csv)
        self.files = sorted(glob.glob(os.path.join(root, "train") + "/*.*"))
        
        if train: # if train, then the files are here at this path
            self.files = sorted(glob.glob(os.path.join(root, "train") + "/*.*"))
            t_list = [transforms.Resize(32),
                      transforms.Grayscale(num_output_channels=1)]
            
        else: # if it's test, then it's under subdir "test"
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
            t_list = [transforms.Resize(32), 
                      transforms.Grayscale(num_output_channels=1)]

        t_list += [transforms.ToTensor()] # all of that plus make it into tensor
        
        self.T_ims = transforms.Compose(t_list)

        # texture paths
        background_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'background'
        self.background_textures = sorted([im for im in background_dir.glob('*.jpg')])
        
        object_dir = Path('.') / 'mnists' / 'data' / 'textures' / 'object'
        self.object_textures = sorted([im for im in object_dir.glob('*.jpg')])

        # transforming the .jpg background and object images
        # there are exactly 10 each of these, for each of th 10 classes
        # eurecom dataset now has 10 labels also
        self.T_texture = transforms.Compose([
            transforms.Resize((inter_sz, inter_sz), Image.NEAREST),
            transforms.RandomCrop(self.mnist_sz, padding=3, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __getitem__(self, index):
        
        # 1 - get image: will open the images in self.files that's declared earlier in __item__ for train or test
        ims = Image.open(self.files[index % len(self.files)]).convert('RGB') # RGB but will be grayscaled to 32 x 32:
        ims = self.T_ims(ims) # 32x32, grayscale, apply transforms from __init__
        ims = F.interpolate(ims[None, :], (self.mnist_sz, self.mnist_sz)).squeeze() #from cfg code
        ims = (ims > 0.1).to(int)  # binarize from cfg code
        print("ims.shape:", ims.shape)
        
        # 2 - get label: retrieve label at that index
        labels = self.annots.iloc[index, 1] #the label is the second col
        #labels = np.array([labels])
        #target = torch.Tensor(target) #don't do this, makes a float
        #labels = torch.LongTensor(labels) #int64
        labels = torch.tensor(labels).to(torch.int64) #from cgn function 'transform_labels(labels)'
        print("labels:", labels)
        print("labels.shape:", labels.shape)
        
        # 3 - get textures: Backgrounds
        i = labels if self.train else np.random.randint(10) #index is the label retrieved like 0,1,3, etc., if test assign random int
        back_text = Image.open(self.background_textures[i])
        back_text = self.T_texture(back_text)

        # 4 - get textures: Objects
        i = labels if self.train else np.random.randint(10)
        obj_text = Image.open(self.object_textures[i])
        obj_text = self.T_texture(obj_text)

        """
        # get digit
        im_digit = (self.ims_digit[idx]/255.).to(torch.float32)  
        im_digit = F.interpolate(im_digit[None, :], (self.mnist_sz, self.mnist_sz)).squeeze()
        im_digit = (im_digit > 0.1).to(int)  # binarize

        # plot digit onto the texture
        ims = im_digit*(obj_text) + (1 - im_digit)*back_text

        ret = {
            'ims': ims,
            'labels': self.labels[idx],
        }
        print("WildlifeMNIST", ret)
        return ret
        """
    
        # plot digit onto the texture
        ims = ims*(obj_text) + (1 - ims)*back_text
        
        ret = {
            'ims': ims,
            'labels': labels,
        }
        return ret

    def __len__(self):
        return len(self.files)
        #return self.labels.shape[0]    
    
    
    
#==========HELPER LOADERS=================================================

"""
# get data
    dataloader, _ = get_dataloaders(cfg.TRAIN.DATASET, cfg.TRAIN.BATCH_SIZE,
                                    cfg.TRAIN.WORKERS)

"""
def get_dataloaders(dataset, batch_size, workers):
    if dataset == 'colored_MNIST':
        MNIST = ColoredMNIST
    elif dataset == 'double_colored_MNIST':
        MNIST = DoubleColoredMNIST
    elif dataset == 'wildlife_MNIST':
        MNIST = WildlifeMNIST
    elif dataset =='Eurecom':
        MNIST = Eurecom
    else:
        raise TypeError(f"Unknown dataset: {dataset}")

    if dataset =='Eurecom':
        dl_train = DataLoader(Eurecom(root = "/home/local/AD/cordun1/experiments/data/Eurecom_Thermal", 
                                           annots_csv = "/home/local/AD/cordun1/experiments/data/labels/Eur_labels2_cfg.csv", train=True),
                                               batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
        dl_test = DataLoader(Eurecom(root = "/home/local/AD/cordun1/experiments/data/Eurecom_Thermal", 
                                      annots_csv = "/home/local/AD/cordun1/experiments/data/labels/Eur_labels2_cfg.csv", train=False),
                                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    
    else:
        ds_train = MNIST(train=True)
        ds_test = MNIST(train=False)
    
        dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True, num_workers=workers)
        dl_test = DataLoader(ds_test, batch_size=batch_size*2,
                         shuffle=False, num_workers=workers)

    return dl_train, dl_test


#=======TENSOR DATASETS AVAILABLE=====================================

TENSOR_DATASETS = ['colored_MNIST', 'colored_MNIST_counterfactual',
                   'double_colored_MNIST', 'double_colored_MNIST_counterfactual',
                   'wildlife_MNIST', 'wildlife_MNIST_counterfactual',
                  'Eurecom', 'Eurecom_counterfactual']

#=====GET TENSOR DATALOADERS============================================

def get_tensor_dataloaders(dataset, batch_size=64):
    assert dataset in TENSOR_DATASETS, f"Unknown datasets {dataset}"

    if 'counterfactual' in dataset:
        tensor = torch.load(f'mnists/data/{dataset}.pth') #mnists/data/Eurecom.pth
        ds_train = TensorDataset(*tensor[:2]) #ds_train converts these into tensors
        dataset = dataset.replace('_counterfactual', '') #use the counterfactuals instead
    else:
        ds_train = TensorDataset(*torch.load(f'mnists/data/{dataset}_train.pth'))
    
    # only the training set contains counterfactuals; test stays as test
    ds_test = TensorDataset(*torch.load(f'mnists/data/{dataset}_test.pth'))

    if 'Eurecom' in dataset:
        dl_train = DataLoader(Eurecom(root = "/home/local/AD/cordun1/experiments/data/Eurecom_Thermal", 
                                           annots_csv = "/home/local/AD/cordun1/experiments/data/labels/Eur_labels2_cfg.csv", train=True),
                                               batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
        dl_test = DataLoader(Eurecom(root = "/home/local/AD/cordun1/experiments/data/Eurecom_Thermal", 
                                      annots_csv = "/home/local/AD/cordun1/experiments/data/labels/Eur_labels2_cfg.csv", train=False),
                                           batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
        
    else:
        dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4,
                          shuffle=True, pin_memory=True)
        dl_test = DataLoader(ds_test, batch_size=batch_size*10, num_workers=4,
                         shuffle=False, pin_memory=True)

    return dl_train, dl_test
