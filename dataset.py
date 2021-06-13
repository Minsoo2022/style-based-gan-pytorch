from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import pickle


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class MultiResolutionDataset_Brats(Dataset):
    def __init__(self, path, transform, resolution=8):


        self.resolution = resolution
        self.transform = transform
        self.pickle_list = glob.glob(os.path.join(path, str(self.resolution), '*'))

        self.pickle_list = sorted(self.pickle_list)

    def __len__(self):
        return len(self.pickle_list)

    def __getitem__(self, index):
        with open(self.pickle_list[index], 'rb') as f:

            #img = pickle.load(f)
            img = pickle.load(f).astype('float32')

        img = self.transform(img)

        return img
