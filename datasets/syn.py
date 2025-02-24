import numpy as np

from collections.abc import Iterator
from typing import Tuple
from torch.utils.data import Dataset


class InterleavingRounds(Iterator):
    """
    Simple iterator that, given an interleaving ratio, returns the current
    round number and if it is an authentic round. Where the interleaving ratiois    is a tuple of (rho_syn, rho_tot).

    syn_only: defines the first n rounds that will exclusively synthetic,
              regardless of the interleaivng ratio
    """

    def __init__(self, rounds=10, ratio=(0, 1), syn_only=None) -> None:
        self.rho_syn, self.rho_tot = ratio
        self.rounds = iter(range(rounds))
        self.syn_only = syn_only

    def __next__(self):
        round = next(self.rounds)

        if self.syn_only and self.syn_only > round:
            return round, False

        is_auth = round % self.rho_tot < self.rho_tot - self.rho_syn

        return round, is_auth


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, num_classes=10):
        super().__init__()
        self.dataset = dataset
        self.indices = np.array(list(indices))
        self.targets = np.array(dataset.targets)[self.indices]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])


def partition_data(train_data, classes=None) -> Tuple[IndexedDataset, IndexedDataset]:
    """
    Given a training set, evenly divide the dataset into two partitions, where
    both partitions have roughly the same number of samples per class

    Assumes that the train_data has the attribute `train_data.targets`
    """
    auth_idxs = []
    syn_idxs = []

    if classes is None:
        classes = set(train_data.targets)

    for label in range(len(classes)):
        # indices of all the samples with this class
        label_idx = np.where(np.array(train_data.targets) == label)[0]
        # compute how many samples have this class
        label_count = len(label_idx)
        # shuffle it just to be sure, but it shouldn't do anything
        np.random.shuffle(label_idx)

        # let's say that the first halve of the samples are authentic, and
        # the second halve are synthetic
        auth_idx = label_idx[: label_count // 2]
        syn_idx = label_idx[label_count // 2 :]

        auth_idxs.extend(auth_idx)
        syn_idxs.extend(syn_idx)

    auth_data = IndexedDataset(train_data, auth_idxs)
    syn_data = IndexedDataset(train_data, syn_idxs)
    return auth_data, syn_data


def balance_auth_dst(auth_dst, dict_users, syn_dst, num_classes=10):
    """
    Returns the partitioning of synthetic dataset that balances out the whole
    dataset (auth + synth together) by compensating for the imbalance in the
    authentic data paritioning
    """

    dict_users_syn = {}
    counts = {}
    dst_sizes = []

    for k in dict_users:
        dict_users_syn[k] = []
        counts[k] = [0] * 10
        for idx in dict_users[k]:
            counts[k][auth_dst[idx][1]] += 1
        dst_sizes.append(sum(counts[k]))

    dst_sizes = np.array(dst_sizes) / sum(dst_sizes)
    class_distrs = {}
    for i in range(num_classes):
        distr = []
        for k in counts:
            distr.append(counts[k][i])
        distr = np.array(distr)

        sol = distr.sum() - distr
        sol = sol / sol.sum()

        # also scale the solution by weights, where the weights are determined
        # by the dataset size of each client
        sol = sol * dst_sizes
        sol = sol / sol.sum()

        class_distrs[i] = sol

        indices = np.where(syn_dst.targets == i)[0]
        samples_per_client = np.random.multinomial(len(indices), sol)
        start = 0
        for k, num_samples in enumerate(samples_per_client):
            dict_users_syn[k].extend(indices[start : start + num_samples])
            start += num_samples

    return dict_users_syn
