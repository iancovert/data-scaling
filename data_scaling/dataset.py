from torch.utils.data import Dataset


class MarginalContributionStackDataset(Dataset):
    '''
    Dataset for averaged marginal contributions.

    Args:
      dataset: dataset of (x, y) pairs.
      labels: dataset of (delta, cardinality) pairs.
    '''

    def __init__(self, dataset: Dataset, labels: Dataset):
        super().__init__()
        assert len(dataset) == len(labels)
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return tuple([*self.dataset[index], *self.labels[index]])
