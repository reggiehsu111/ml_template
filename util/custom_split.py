from torch._utils import _accumulate
from torch import randperm

def custom_subset(baseDataset, opt):

    class Subset(baseDataset):
        """
        Subset of a dataset at specified indices.

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
        """
        def __init__(self, dataset, indices, opt):
            super().__init__(opt)
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    return Subset



def random_split(dataset, lengths, baseDataset, opt):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    Subset = custom_subset(baseDataset, opt)
    return [Subset(dataset, indices[offset - length:offset], opt) for offset, length in zip(_accumulate(lengths), lengths)]