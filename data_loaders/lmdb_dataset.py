import json

import cv2
import lmdb
import numpy as np
from skimage import color
from torch.utils.data import Dataset, DataLoader



class lmdbDataset(Dataset):

    def __init__(self, lmdb_path, selection_file=None, transform=None):
        self.env = lmdb.open(lmdb_path,
                             max_readers=100,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.indices = range(self.nSamples)
        self.transform = transform
        self.selections = None
        if selection_file is not None:
            self.selections = []
            with open(selection_file) as f:
                for selection in f:
                    self.selections.append(int(selection.strip()))

    def __len__(self):
        if self.selections is not None:
            return len(self.selections)
        return self.nSamples

    def __getitem__(self, index):
        # assert index <= len(self), 'index range error'
        if self.selections is not None:
            index = self.selections[index]
        jsonKey = 'json-%09d' % (index)
        sample = self.txn.get(jsonKey.encode())
        sample = json.loads(sample)
        for k in sample.keys():
            if type(sample[k]) is list:
                sample[k] = np.array(sample[k])

        if self.transform:
            sample = self.transform(sample)

        if 'name' in sample:
            sample['filename'] = sample['name']
        if sample['warp_matrix'].ndim == 3:
            assert np.array_equal(sample['warp_matrix'][-1], np.array([[1, 0, 0], [0, 1, 0]]))
        else:
            assert np.array_equal(sample['warp_matrix'][-1, 0], np.array([[1, 0, 0], [0, 1, 0]]))
        return sample


def lmdb_nsamples(db):
    env = lmdb.open(db,
                    max_readers=1,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
    return nSamples


if __name__ == "__main__":
    ds = lmdbDataset('data/WD.lmdb', 'data/Waterloo_Dataset/val.txt')
    print(len(ds))
    dataloader_val = DataLoader(ds, batch_size=24, num_workers=5,
                                shuffle=True, pin_memory=False)
    for i in dataloader_val:
        continue
