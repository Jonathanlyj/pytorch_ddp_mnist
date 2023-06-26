from typing import Tuple

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import pncpy
import struct
from array import array

DISABLE_TQDM = True


# torch dataloader
class MNISTNetCDF(Dataset):
    
    def __init__(self, root_dir, is_train=True, transforms=None, comm=None):

        if is_train:
            labels_filepath = os.path.join(root_dir,'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        else:
            labels_filepath = os.path.join(root_dir,'t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        self.labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            self.labels = array("B", file.read())  
    
        self.transforms = transforms
        print ('=> Reading NetCDF File...')
        nc_path = os.path.join(root_dir,'mnist_{}_images.nc'.format('train' if is_train else 'test'))
        self.nc = pncpy.File(nc_path,'r')
        print('=> Dataset created, image nc file is : {}'.format(nc_path))
        
    def __len__(self):
        return self.nc.variables['images'].shape[0]
    
    def __getitem__(self,index):
        
        # read image 
        # image = np.array(self.nc.variables['images'][index])
        image = np.array(self.nc.variables['images'][index])
        # fetch and encode label
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        
        return image,label



def create_data_loaders(dataset, batch_size: int, num_worker: int,) -> Tuple[DataLoader, DataLoader]:

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_worker,
                        persistent_workers=True if num_worker > 0 else False)

    # This is not necessary to use distributed sampler for the test or validation sets.
    return loader


def create_model():
    # create model architecture
    model = nn.Sequential(
        nn.Linear(28*28, 128),  # MNIST images are 28x28 pixels
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)  # 10 classes to predict
    )
    return model


def main(epochs: int,
         model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader):


    # train the model
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        # train the model for one epoch
        for index, (x, y) in enumerate(train_loader):
            if index %500 == 0:
                print(x[0])
            
    return 



#     batch_size = 128
#     epochs = 1
#     train_loader, test_loader = create_data_loaders(batch_size, 1)
#     main(epochs=epochs,
#          model=create_model(),
#          train_loader=train_loader,
#          test_loader=test_loader)
if __name__ == '__main__':
    batch_size = 128
    epochs = 1
    num_worker = 2
    dataset_loc = '.'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=True,
                                transforms=transform)

    test_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=False,
                                transforms=transform)

    # train_loader = create_data_loaders(train_dataset, batch_size, 2)
    # test_loader = create_data_loaders(test_dataset, batch_size, 2)

    train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_worker,
                        persistent_workers=True if num_worker > 0 else False)

    test_loader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_worker,
                        persistent_workers=True if num_worker > 0 else False)

    main(epochs=epochs,
        model=create_model(),
        train_loader=train_loader,
        test_loader=test_loader)