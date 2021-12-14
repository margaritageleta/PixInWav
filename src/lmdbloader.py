import os
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
from loader import StegoDataset

MY_FOLDER = '/mnt/gpid07/imatge/teresa.domenech/venv/PixInWav'
DATA_FOLDER = '/projects/deep_learning/ILSVRC/ILSVRC2012'
AUDIO_FOLDER = '/mnt/gpid08/users/teresa.domenech'

def make_lmdb(dataset, save_dir=f'{MY_FOLDER}/data', num_workers=4):
    loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])

    lmdb_file = os.path.join(save_dir, 'dataset.lmdb')
    env = lmdb.open(lmdb_file, map_size=1099511627776)

    txn = env.begin(write=True)
    for index, (sample, target) in enumerate(loader):
        obj = (sample.tobytes(), sample.shape, target.tobytes())
        txn.put(f'{index:06d}'.encode(), pickle.dumps(obj))
        if index % 10000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    env.sync()
    env.close()

    return lmdb_file


class LMDBDataset(Dataset):
    def __init__(self, lmdb_file, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.env = lmdb.open(lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            data = txn.get(f'{index:06d}'.encode())

        sample_bytes, sample_shape, target_bytes = pickle.loads(data)
        sample = np.fromstring(sample_bytes, dtype=np.uint8).reshape(sample_shape)
        target = np.fromstring(target_bytes, dtype=np.float32)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    mappings = {}
    with open(f'{MY_FOLDER}/data/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img

    dataset = StegoDataset(image_root=DATA_FOLDER,
                           audio_root=AUDIO_FOLDER,
                           folder='train',
                           mappings=mappings)
    print('Dataset prepared')
    lmdb_file = make_lmdb(dataset)
    lmdb_dataset = LMDBDataset(lmdb_file)