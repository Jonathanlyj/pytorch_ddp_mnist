from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm

DISABLE_TQDM = True


def create_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.MNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # This is not necessary to use distributed sampler for the test or validation sets.
    test_dataset = datasets.MNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader


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
         test_loader: DataLoader) -> nn.Module:
    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        # train the model for one epoch

        for idx, (x, y) in enumerate(train_loader):
            x = x.view(x.shape[0], -1)
            # optimizer.zero_grad()
            # y_hat = model(x)
            # batch_loss = loss(y_hat, y)
            # batch_loss.backward()
            # optimizer.step()
            # batch_loss_scalar = batch_loss.item()
            # epoch_loss += batch_loss_scalar / x.shape[0]
            if idx % 500 == 0:
                print(x[0])
                print(y)
           

        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for idx, (x, y) in enumerate(test_loader):
                if idx % 500 == 0:
                    print(x[0])
                    print(y)

    return 


if __name__ == '__main__':
    batch_size = 128
    epochs = 1
    train_loader, test_loader = create_data_loaders(batch_size)
    main(epochs=epochs,
                model=create_model(),
                train_loader=train_loader,
                test_loader=test_loader)
