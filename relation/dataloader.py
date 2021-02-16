import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from utils.dataset import OmniglotDataset, MiniImagenetDataset, get_data_dir

DATADIR = get_data_dir()


def get_dataloader(args, *modes):
    res = []
    if 'train' in modes[0]:
        print("Loading data...", end='')
    for mode in modes:
        if args.dataset == 'omniglot':
            mdb_path = os.path.join(DATADIR, 'relation_mdb', 'omniglot_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except FileNotFoundError:
                dataset = OmniglotDataset(mode)
                if not os.path.exists(os.path.dirname(mdb_path)):
                    os.makedirs(os.path.dirname(mdb_path))
                torch.save(dataset, mdb_path)

        elif args.dataset == 'miniImagenet':
            mdb_path = os.path.join(DATADIR, 'relation_mdb', 'miniImagenet_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except FileNotFoundError:
                dataset = MiniImagenetDataset(mode)
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

        sampler = RelationBatchSampler(dataset.y, classes_per_it, num_support, num_query, args.episodes)
        data_loader = DataLoader(dataset, batch_sampler=sampler,
                                 pin_memory=True if torch.cuda.is_available() else False)
        res.append(data_loader)

    if 'train' in modes[0]:
        print("Loading data...", end='')
    if len(modes) == 1:
        return res[0]
    else:
        return res


class RelationBatchSampler(Sampler):
    def __init__(self, labels, classes_per_it, num_support, num_query, episodes, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_support = num_support
        self.num_query = num_query
        self.episodes = episodes

        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx

    def __iter__(self):
        """
        yield a batch of indexes
        """
        ns = self.num_support
        nq = self.num_query
        cpi = self.classes_per_it
        counts = self.counts[0].item()

        for _ in range(self.episodes):
            batch_s = torch.LongTensor(ns * cpi)
            batch_q = torch.LongTensor(nq * cpi)
            classes_idxs = torch.randperm(len(self.classes))[:cpi]  # 랜덤으로 클래스 60개 선택
            for i, c in enumerate(self.classes[classes_idxs]):
                s_s = slice(i * ns, (i + 1) * ns)  # 하나의 클래스당 선택한 support 이미지
                s_q = slice(i * nq, (i + 1) * nq)  # 하나의 클래스당 선택한 query 이미지

                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(counts)[:ns + nq]

                batch_s[s_s] = self.indexes[label_idx][sample_idxs][:ns]
                batch_q[s_q] = self.indexes[label_idx][sample_idxs][ns:]
            batch = torch.cat((batch_s, batch_q))
            yield batch

    def __len__(self):
        """
        returns the number of episodes (episodes) per epoch
        """
        return self.episodes
