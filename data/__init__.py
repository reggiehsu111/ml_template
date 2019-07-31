import importlib
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from util.custom_split import random_split
from .base_dataset import *


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataloader(opt):
    """Create a dataloader given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataloader(opt)
    """
    dataset_class = find_dataset_using_name(opt.args.dataset_mode)
    
    dataset = dataset_class(opt)
    print("dataset [%s] was created" % type(dataset).__name__)

    # Randomly split dataset into training, validation and test sets
    train_size = int(opt.args.train_val_test[0]*len(dataset))
    val_size = int(opt.args.train_val_test[1]*len(dataset))
    test_size = len(dataset) - train_size - val_size

    # custom random_split() from custom_split.py
    train_DataSet, val_DataSet, test_DataSet = random_split(dataset, [train_size, val_size, test_size], type(dataset), opt)
    print("type of splitted datasets: ", type(train_DataSet))

    # Create dataloader for each phase
    train_dataloader = CustomDatasetDataLoader(opt,train_DataSet)
    val_dataloader = CustomDatasetDataLoader(opt,val_DataSet)
    test_dataloader = CustomDatasetDataLoader(opt,test_DataSet)

    # initiate each dataloader
    train_dataloader = train_dataloader.load_data()
    val_dataloader = val_dataloader.load_data()
    test_dataloader = test_dataloader.load_data()

    return [train_dataloader, val_dataloader, test_dataloader]

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataset):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.args = opt.args
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=not self.args.serial_batches,
            num_workers=int(self.args.num_workers),
            # use custom collate_fn if defined
            collate_fn= default_collate if not hasattr(self.dataset, 'collate_fn') else self.dataset.collate_fn
            )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.args.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.args.batch_size >= self.args.max_dataset_size:
                break
            yield data

    def __getitem__(self, index):
        assert float(index) < self.args.max_dataset_size
        return self.dataset[index]
