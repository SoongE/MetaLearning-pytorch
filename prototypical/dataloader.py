import os

import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np

from utils.dataset import OmniglotDataset, MiniImageNetDataset, get_data_dir

warnings.filterwarnings("ignore")

DATADIR = get_data_dir()


def get_dataloader(args, *modes):
    res = []
    if 'train' in modes[0]:
        print("Loading data...", end='')
    for mode in modes:
        if args.dataset == 'omniglot':
            mdb_path = os.path.join(DATADIR, 'proto_mdb', 'omniglot_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except FileNotFoundError:
                dataset = OmniglotDataset(mode)
                if not os.path.exists(os.path.dirname(mdb_path)):
                    os.makedirs(os.path.dirname(mdb_path))
                torch.save(dataset, mdb_path)

        elif args.dataset == 'miniImageNet':
            mdb_path = os.path.join(DATADIR, 'proto_mdb', 'miniImageNet_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except FileNotFoundError:
                dataset = MiniImageNetDataset(mode)
                if not os.path.exists(os.path.dirname(mdb_path)):
                    os.makedirs(os.path.dirname(mdb_path))
                torch.save(dataset, mdb_path)

        if 'train' in mode:
            classes_per_it = args.classes_per_it_tr
            num_support = args.num_support_tr
            num_query = args.num_query_tr
        else:
            classes_per_it = args.classes_per_it_val
            num_support = args.num_support_val
            num_query = args.num_query_val

        sampler = PrototypicalBatchSampler(dataset.y, classes_per_it, num_support, num_query, args.iterations)
        data_loader = DataLoader(dataset, batch_sampler=sampler,
                                 pin_memory=True if torch.cuda.is_available() else False)
        res.append(data_loader)

    if 'train' in modes[0]:
        print("Loading data...", end='')
    if len(modes) == 1:
        return res[0]
    else:
        return res


class PrototypicalBatchSampler(Sampler):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, classes_per_it, num_samples_support, num_samples_query, iterations, data_source=None):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super().__init__(data_source)
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_samples_support = num_samples_support
        self.num_samples_query = num_samples_query
        self.iterations = iterations

        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """
        yield a batch of indexes
        """
        nss = self.num_samples_support
        nsq = self.num_samples_query
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            batch_s = torch.LongTensor(nss * cpi)
            batch_q = torch.LongTensor(nsq * cpi)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 랜덤으로 클래스 60개 선택
            for i, c in enumerate(self.classes[c_idxs]):
                s_s = slice(i * nss, (i + 1) * nss)  # 하나의 클래스당 선택한 support 이미지
                s_q = slice(i * nsq, (i + 1) * nsq)  # 하나의 클래스당 선택한 query 이미지

                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:nss + nsq]

                batch_s[s_s] = self.indexes[label_idx][sample_idxs][:nss]
                batch_q[s_q] = self.indexes[label_idx][sample_idxs][nss:]
            batch = torch.cat((batch_s, batch_q))
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations
