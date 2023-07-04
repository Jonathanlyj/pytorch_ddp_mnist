import argparse
from typing import Tuple
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import datasets, transforms
from mpi4py import MPI
import torch.distributed as dist
import os
import pncpy
import numpy as np

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
        self.nc.begin_indep()
        
        print('=> Dataset created, image nc file is : {}'.format(nc_path))
        
    def __len__(self):
        return self.nc.variables['images'].shape[0]
    
    def __getitem__(self,index):
        
        # read image 
        # image = np.array(self.nc.variables['images'][index])
        image = np.array(self.nc.variables['images'][index])
        # fetch and encode label
        buff = np.empty((), np.uint8)
        self.nc.variables['labels'].get_var(buff, index = (index,))
        if self.transforms:
            image = self.transforms(image)
        return image,buff

class distributed():
    def get_size(self):
        if dist.is_available() and dist.is_initialized():
            size = dist.get_world_size()
        else:
            size = 1
        return size

    def get_rank(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        return rank

    def get_local_rank(self):
        if not (dist.is_available() and dist.is_initialized()):
            return 0
        # Number of GPUs per node
        if torch.cuda.is_available():
            local_rank = dist.get_rank() % torch.cuda.device_count()
        else:
            # raise NotImplementedError()
            # running on cpu device should not call this function
            local_rank = -1
        return local_rank

    def __init__(self, method):
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function
    
        if method == "nccl-slurm":
            # MASTER_ADDR can be set in the slurm batch script using command
            # scontrol show hostnames $SLURM_JOB_NODELIST
            if "MASTER_ADDR" not in os.environ:
                # Try SLURM_LAUNCH_NODE_IPADDR but it is the IP address of the node
                # from which the task launch was initiated (where the srun command
                # ran from). It may not be the node of rank 0.
                if "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
                    os.environ["MASTER_ADDR"] = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
                else:
                    raise Exception("Error: nccl-slurm - SLURM_LAUNCH_NODE_IPADDR is not set")
    
            # Use the default pytorch port
            if "MASTER_PORT" not in os.environ:
                if "SLURM_SRUN_COMM_PORT" in os.environ:
                    os.environ["MASTER_PORT"] = os.environ["SLURM_SRUN_COMM_PORT"]
                else:
                    os.environ["MASTER_PORT"] = "29500"
    
            # obtain WORLD_SIZE
            if "WORLD_SIZE" not in os.environ:
                if "SLURM_NTASKS" in os.environ:
                    world_size = os.environ["SLURM_NTASKS"]
                else:
                    if "SLURM_JOB_NUM_NODES" in os.environ:
                        num_nodes = os.environ["SLURM_JOB_NUM_NODES"]
                    else:
                        raise Exception("Error: nccl-slurm - SLURM_JOB_NUM_NODES is not set")
                    if "SLURM_NTASKS_PER_NODE" in os.environ:
                        ntasks_per_node = os.environ["SLURM_NTASKS_PER_NODE"]
                    elif "SLURM_TASKS_PER_NODE" in os.environ:
                        ntasks_per_node = os.environ["SLURM_TASKS_PER_NODE"]
                    else:
                        raise Exception("Error: nccl-slurm - SLURM_(N)TASKS_PER_NODE is not set")
                    world_size = ntasks_per_node * num_nodes
                os.environ["WORLD_SIZE"] = str(world_size)
    
            # obtain RANK
            if "RANK" not in os.environ:
                if "SLURM_PROCID" in os.environ:
                    os.environ["RANK"] = os.environ["SLURM_PROCID"]
                else:
                    raise Exception("Error: nccl-slurm - SLURM_PROCID is not set")
    
            # Initialize DDP module
            dist.init_process_group(backend = "nccl", init_method='env://')
    
        elif method == "nccl-openmpi":
            if "MASTER_ADDR" not in os.environ:
                if "PMIX_SERVER_URI2" in os.environ:
                    os.environ["MASTER_ADDR"] = os.environ("PMIX_SERVER_URI2").split("//")[1]
                else:
                    raise Exception("Error: nccl-openmpi - PMIX_SERVER_URI2 is not set")
    
            # Use the default pytorch port
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
    
            if "WORLD_SIZE" not in os.environ:
                if "OMPI_COMM_WORLD_SIZE" not in os.environ:
                    raise Exception("Error: nccl-openmpi - OMPI_COMM_WORLD_SIZE is not set")
                os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    
            if "RANK" not in os.environ:
                if "OMPI_COMM_WORLD_RANK" not in os.environ:
                    raise Exception("Error: nccl-openmpi - OMPI_COMM_WORLD_RANK is not set")
                os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    
            # Initialize DDP module
            dist.init_process_group(backend = "nccl", init_method='env://')
    
        elif method == "nccl-mpich":
            if "MASTER_ADDR" not in os.environ:
                os.environ['MASTER_ADDR'] = "localhost"
    
            # Use the default pytorch port
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
    
            if "WORLD_SIZE" not in os.environ:
                if "PMI_SIZE" in os.environ:
                    world_size = os.environ["PMI_SIZE"]
                elif MPI.Is_initialized():
                    world_size = MPI.COMM_WORLD.Get_size()
                else:
                    world_size = 1
                os.environ["WORLD_SIZE"] = str(world_size)
    
            if "RANK" not in os.environ:
                if "PMI_RANK" in os.environ:
                    rank = os.environ["PMI_RANK"]
                elif MPI.Is_initialized():
                    rank = MPI.COMM_WORLD.Get_rank()
                else:
                    rank = 0
                os.environ["RANK"] = str(rank)
    
            # Initialize DDP module
            dist.init_process_group(backend = "nccl", init_method='env://')
        
        elif method ==  "mpich":
            if "MASTER_ADDR" not in os.environ:
                os.environ['MASTER_ADDR'] = "localhost"
    
            # Use the default pytorch port
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
    
            if "WORLD_SIZE" not in os.environ:
                if "PMI_SIZE" in os.environ:
                    world_size = os.environ["PMI_SIZE"]
                elif MPI.Is_initialized():
                    world_size = MPI.COMM_WORLD.Get_size()
                else:
                    world_size = 1
                os.environ["WORLD_SIZE"] = str(world_size)
    
            if "RANK" not in os.environ:
                if "PMI_RANK" in os.environ:
                    rank = os.environ["PMI_RANK"]
                elif MPI.Is_initialized():
                    rank = MPI.COMM_WORLD.Get_rank()
                else:
                    rank = 0
                os.environ["RANK"] = str(rank)
    
            # Initialize DDP module
            dist.init_process_group(backend = "mpi", init_method='env://')
    
        elif method == "gloo":
            if "MASTER_ADDR" not in os.environ:
                # check if OpenMPI is used
                if "PMIX_SERVER_URI2" in os.environ:
                    addr = os.environ["PMIX_SERVER_URI2"]
                    addr = addr.split("//")[1].split(":")[0]
                    os.environ["MASTER_ADDR"] = addr
                else:
                    os.environ['MASTER_ADDR'] = "localhost"
    
            # Use the default pytorch port
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
    
            # obtain WORLD_SIZE
            if "WORLD_SIZE" not in os.environ:
                # check if OpenMPI is used
                if "OMPI_COMM_WORLD_SIZE" in os.environ:
                    world_size = os.environ["OMPI_COMM_WORLD_SIZE"]
                elif "PMI_SIZE" in os.environ:
                    world_size = os.environ["PMI_SIZE"]
                elif MPI.Is_initialized():
                    world_size = MPI.COMM_WORLD.Get_size()
                else:
                    world_size = 1
                os.environ["WORLD_SIZE"] = str(world_size)
    
            # obtain RANK
            if "RANK" not in os.environ:
                # check if OpenMPI is used
                if "OMPI_COMM_WORLD_RANK" in os.environ:
                    rank = os.environ["OMPI_COMM_WORLD_RANK"]
                elif "PMI_RANK" in os.environ:
                    rank = os.environ["PMI_RANK"]
                elif MPI.Is_initialized():
                    rank = MPI.COMM_WORLD.Get_rank()
                else:
                    rank = 0
                os.environ["RANK"] = str(rank)
    
            # Initialize DDP module
            dist.init_process_group(backend = "gloo", init_method='env://')
    
        else:
            raise NotImplementedError()
    
    def reduceMAX(self, src, root):
        # dist.reduce(src, root, dist.ReduceOp.MAX)
        # return src.cpu().numpy()
        import numpy
        dst = numpy.empty(len(src))
        MPI.COMM_WORLD.Reduce(src, dst, op=MPI.MAX, root=root)
        return dst
    
    def barrier(self):
        MPI.COMM_WORLD.Barrier()
        # dist.barrier()
    
    def finalize(self):
        dist.destroy_process_group()

def configure():
    # Configuration options (overwrite default configuration with command-line arguments)
    parser = argparse.ArgumentParser(description="Evaluate cost of reading input files")
    add_arg = parser.add_argument
    add_arg("--wireup_method", type=str, default="nccl-slurm",
        choices=["nccl-slurm", "nccl-openmpi", "nccl-mpich", "gloo", "mpich"],
        help="Choose backend for distributed environment initialization")
    add_arg("--data_path",   type=str, default=None, help="File path to training samples")
    add_arg("--data_limit",  type=int, default=None, help="Max number of samples to be used")
    add_arg("--batch_size",  type=int, default=None, help="Batch size")
    add_arg("--n_epochs",    type=int, default=None, help="Number of epochs")
    add_arg("--num_workers", type=int, default=None, help="Number of subprocesses to use for data loading")
    add_arg("--parallel",    action='store_true',    help="To run in parallel")
    add_arg("--hdf5",        action='store_true',    help="Read from HDF5 files")

    args = parser.parse_args()

    config = {}
    config["trainer"] = {}
    config["data"] = {}
    config["trainer"]["batch_size"]    = 128
    config["trainer"]["wireup_method"] = args.wireup_method
    config["trainer"]["parallel"]      = args.parallel
    config["trainer"]["device"]        = 0
    config["trainer"]["n_epochs"]      = 1
    config["trainer"]["num_workers"]   = 0
    config["data"]["limit"]            = None
    config["data"]["label_map"]        = [ 0, 1, 0, 0, 2, 3, 1, 4 ]
    config["data"]["hdf5"]             = args.hdf5
    if args.data_path   != None: config["data"]["path"]           = args.data_path
    if args.data_limit  != None: config["data"]["limit"]          = args.data_limit
    if args.batch_size  != None: config["trainer"]["batch_size"]  = args.batch_size
    if args.n_epochs    != None: config["trainer"]["n_epochs"]    = args.n_epochs
    if args.num_workers != None: config["trainer"]["num_workers"] = args.num_workers

    return config

def init_parallel(config):
    # check if cuda device is available
    ngpu_per_node = torch.cuda.device_count()
    if not torch.cuda.is_available():
        config["trainer"]["wireup_method"] = "gloo"
        config["trainer"]["device"] = "cpu"

    rank = 0
    world_size = 1
    # initialize parallel/distributed environment
    if config["trainer"]["parallel"]:
        comm = distributed(config["trainer"]["wireup_method"])
        rank = comm.get_rank()
        world_size = comm.get_size()
        # TODO: must check whether num_workers > 0 works in DDP
        # config["trainer"]["num_workers"] = 0
        # ignoring config["trainer"]["device"]
        local_rank = comm.get_local_rank()
    else:
        comm = None
        local_rank = config["trainer"]["device"]

    # select training device: cpu or cuda
    if config["trainer"]["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:"+str(local_rank))

    config["trainer"]["device"]     = device
    config["trainer"]["rank"]       = rank
    config["trainer"]["world_size"] = world_size

    # Print out the settings and init timings
    if rank == 0:
        import socket
        print('------------------------------------------------------------------')
        if config["trainer"]["parallel"]:
            print('\n======== I/O evaluation in Parallel GNN Training ========')
        else:
            print('\n======== I/O evaluation in Serial GNN Training ========')
        print("%-32s: %s" % ('Host name',socket.gethostname()))
        print("%-32s: %d" % ('Number of processes', world_size))
        print("%-32s: %d" % ('number of GPUs per node',ngpu_per_node))
        if device.type == 'cuda':
            print("%-32s: %s" % ('Rank 0 GPU device',device))
        else:
            print('Rank 0 is Using CPU device')
        # print("%-32s: %s" % ('Input file path', config["data"]["path"]))
        if config["data"]["hdf5"]:
            print("%-32s: %s" % ('Input file format', 'A single HDF5 file'))
        else:
            print("%-32s: %s" % ('Input file format', 'Pytorch files, one per sample'))
        print("%-32s: %d" % ('DataLoader num_workers',config["trainer"]["num_workers"]))
        print("%-32s: %d" % ('Number of epochs', config["trainer"]["n_epochs"]))
        print('------------------------------------------------------------------')

    return comm


def create_data_loaders(config, comm) -> Tuple[DataLoader, DataLoader]:
    batch_size = config["trainer"]["batch_size"]
    rank = config["trainer"]["rank"] 
    world_size = config["trainer"]["world_size"] 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_loc = '.'
    train_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=True,
                                transforms=transform,
                                comm  = comm)

    test_dataset = MNISTNetCDF(root_dir=dataset_loc,
                                is_train=False,
                                transforms=transform,
                                comm = comm)

    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world_size,  # Number of GPUs
                                 rank=rank,  # GPU where process is running
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=42)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=0,
                              sampler=sampler,
                              pin_memory=True)
    # This is not necessary to use distributed sampler for the test or validation sets.
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
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


def main(model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader,
         config) -> nn.Module:
    rank  = config["trainer"]["rank"]

    # if config["trainer"]["device"] == "cpu":
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device(f'cuda:{rank}')
    device = torch.device("cpu")
    epochs = config["trainer"]["n_epochs"]

    model = model.to(device)
    model = DistributedDataParallel(model)
    # model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    # train the model
    for i in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(i)

        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                batch_loss_scalar = batch_loss.item()

                val_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

    return model.module


if __name__ == '__main__':


    config = configure()
    mpi_comm = MPI.COMM_WORLD

    # initialize parallel environment
    comm = init_parallel(config)
    rank = config["trainer"]["rank"] 

    # torch.cuda.service(rank)
    # torch.distributed.init_process_group(backend=Backend.NCCL,
    #                                      init_method='env://')

    # mpi_comm = torch.distributed.distributed_c10d._get_default_group().nccl_comm
    train_loader, test_loader = create_data_loaders(config, mpi_comm)
    model = main(model=create_model(),
                 train_loader=train_loader,
                 test_loader=test_loader,
                 config = config)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')
