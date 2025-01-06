import torch
from torchvision import transforms
from torch.utils.data import Dataset

import cv2
from PIL import Image
import numpy as np
import os
import glob


class ForamDatasetPreLoaded(Dataset):

    def __init__(self, targets, file_names, domain, transform=None):
        self.targets = []
        self.file_names = file_names
        self.transform = transform
        self.domain = domain

        self.image = []
        for file in self.file_names:
            image = Image.open(file).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.image.append(image)

        if domain == "real":
            for file in self.file_names:
                label = file.split(os.sep)[-3]
                self.targets.append(targets[label])


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        X = self.image[idx]
        if self.domain == 'real':
            targets = self.targets[idx]
        else:
            targets = 1
        return (X,targets)
    
    def class_distribution(self):
        class_dict = {}
        for file in self.file_names:
            path = file.split(os.sep)[-3]
            if path not in class_dict:
                class_dict[path] = 1
            else:
                class_dict[path] += 1
        
        for k, v in class_dict.items():
            print(k, ":", v)


class ForamDataset(Dataset):

    def __init__(self, targets, file_names, domain, transform=None):
        self.targets = targets
        self.file_names = file_names
        self.transform = transform
        self.domain = domain

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        label_name = self.file_names[idx].split(os.sep)[-3]
        image = Image.open(self.file_names[idx]).convert('RGB')
        if self.domain == 'real':
            targets = self.targets[label_name]
        else:
            targets = 1
        if self.transform:
            X = self.transform(image)
        return (X,targets)
    
    def class_distribution(self):
        class_dict = {}
        for file in self.file_names:
            path = file.split(os.sep)[-3]
            if path not in class_dict:
                class_dict[path] = 1
            else:
                class_dict[path] += 1
        
        for k, v in class_dict.items():
            print(k, ":", v)


def load_dataset(opt, path, **kwargs):

    if 'img_format' in kwargs:
        files_names = glob.glob(os.path.join(path,'*\\images\\*{}'.format(kwargs['img_format'])))
    else:
        raise ValueError('Image format to read not specified.')

    if 'domain' not in kwargs:
        raise ValueError('Domain of data not specified.')

    if kwargs['domain'] == 'real':
        mean = np.array([0.10755069, 0.11711329, 0.12787095])
        std = np.array([0.17114662, 0.18693903, 0.20360334])

    elif kwargs['domain'] == 'synthetic':
        mean = np.array([0.2543242 , 0.25700261, 0.26067707])
        std = np.array([0.06849919, 0.06680629, 0.07529792])
        # train_len = kwargs['train_len']
        # files = files_names
        # files_names = []
        # for i in range(train_len // len(files)):
        #     files_names += files

    else:
        data = []
        for i in files_names:
            data.append(cv2.resize(cv2.imread(i),(opt.image_size,opt.image_size)))
        data = np.array(data)
        mean = data.mean(axis=(0,1,2))/255
        std = data.std(axis=(0,1,2))/255
        data = []

    dataset = ForamDatasetPreLoaded(\
                                  opt.label_dict, files_names,kwargs['domain'],
                                  transforms.Compose([\
                                                        transforms.Resize((opt.image_size,opt.image_size)),\
                                                        # transforms.RandomHorizontalFlip(),\
                                                        transforms.ToTensor(),\
                                                        transforms.Normalize((mean), (std))\
                                                        
                                                    ])
                                )

    return dataset

if __name__ == "__main__":
    from opt import opts
    opt = opts().parse()
    load_dataset(opt, opt.train_synthetic, img_format = 'png', domain = 'synthetic')
    
