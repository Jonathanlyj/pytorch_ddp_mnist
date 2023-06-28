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
from mpi4py import MPI

DISABLE_TQDM = True


# torch dataloader
class MNISTNetCDF(Dataset):
    
    def __init__(self, root_dir, is_train=True, transforms=None, comm=None):

        if is_train:
            labels_filepath = os.path.join(root_dir,'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        else:
            labels_filepath = os.path.join(root_dir,'t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        
    
        self.transforms = transforms
        print ('=> Reading NetCDF File...')
        nc_path = os.path.join(root_dir,'mnist_{}_images.nc'.format('train' if is_train else 'test'))
        self.nc = pncpy.File(nc_path,'r', comm = comm)
        
        print('=> Dataset created, image nc file is : {}'.format(nc_path))
        
    def __len__(self):
        return self.nc.variables['images'].shape[0]
    
    def __getitem__(self,index):
        
        # read image 
        # image = np.array(self.nc.variables['images'][index])
        image = np.array(self.nc.variables['images'][index])
        # fetch and encode label
        buff = np.empty((), np.uint8)
        self.nc.variables['labels'].get_var_all(buff, index = (index,))
        if self.transforms:
            image = self.transforms(image)
        return image,buff



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
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    # train the model
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')

        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            pbar = tqdm(test_loader)
            for x, y in pbar:
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                batch_loss_scalar = batch_loss.item()

                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    batch_size = 128
    epochs = 1
    num_worker = 4
    dataset_loc = '.'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=True,
                                transforms=transform,
                                comm  = comm)

    test_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=False,
                                transforms=transform,
                                comm = comm)

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