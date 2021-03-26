import numpy as np
from PIL import Image, ImageColor
from pathlib import Path

import torch
import torch.nn.functional as F

from torch import tensor
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset


#=========

class Eurecom(Dataset):
    # Adopted for my Eurecom dataset
    
    def __init__(self, train=True, annots_csv, root):
        self.train = train
        self.mnist_sz = 32
        inter_sz = 150
        self.annots = pd.read_csv(annots_csv)
        
        if train: # if train, then the files are here at this path
            self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
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
    
