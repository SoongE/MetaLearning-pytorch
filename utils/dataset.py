import os
import pickle
import shutil
import warnings
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision import datasets as dset

import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_data_dir():
    return os.path.join(BASEDIR, 'data')


class MiniImagenetDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.root_dir = BASEDIR + '/data/miniImagenet'

        if not os.path.exists(self.root_dir):
            print('Data not found. Downloading data')
            self.download()

        dataset = pickle.load(open(os.path.join(self.root_dir, mode), 'rb'))

        self.x = dataset['image_data']

        self.y = torch.arange(len(self.x))
        for idx, (name, id) in enumerate(dataset['class_dict'].items()):
            s = slice(id[0], id[-1] + 1)
            self.y[s] = idx

    def __getitem__(self, index):

        img = self.x[index]

        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        x = transform(image=img)['image']

        return x.permute(0, 1, 2), self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        import tarfile
        gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
        gz_filename = 'mini-imagenet.tar.gz'
        root = BASEDIR + '/data/miniImagenet'

        self.download_file_from_google_drive(gdrive_id, root, gz_filename)

        filename = os.path.join(root, gz_filename)

        with tarfile.open(filename, 'r') as f:
            f.extractall(root)

        os.rename(BASEDIR + '/data/miniImagenet/mini-imagenet-cache-train.pkl', BASEDIR + '/data/miniImagenet/train')
        os.rename(BASEDIR + '/data/miniImagenet/mini-imagenet-cache-val.pkl', BASEDIR + '/data/miniImagenet/val')
        os.rename(BASEDIR + '/data/miniImagenet/mini-imagenet-cache-test.pkl', BASEDIR + '/data/miniImagenet/test')

    def download_file_from_google_drive(self, file_id, root, filename):
        from torchvision.datasets.utils import _get_confirm_token, _save_response_content

        """Download a Google Drive file from  and place it in root.
        Args:
            file_id (str): id of file to be downloaded
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the id of the file.
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
        import requests
        url = "https://docs.google.com/uc?export=download"

        root = os.path.expanduser(root)
        if not filename:
            filename = file_id
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        if os.path.isfile(fpath):
            print('Using downloaded and verified file: ' + fpath)
        else:
            session = requests.Session()

            response = session.get(url, params={'id': file_id}, stream=True)
            token = _get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(url, params=params, stream=True)

            _save_response_content(response, fpath)


class OmniglotDataset(Dataset):
    def __init__(self, mode='trainval'):
        super().__init__()
        self.root_dir = BASEDIR + '/data/omniglot'
        self.vinyals_dir = BASEDIR + '/data/vinyals/omniglot'

        if not os.path.exists(self.root_dir):
            print('Data not found. Downloading data')
            self.download()

        self.x, self.y, self.class_to_idx = self.make_dataset(mode)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        origin_dir = BASEDIR + '/data/omniglot-py'
        processed_dir = self.root_dir

        dset.Omniglot(root=BASEDIR + '/data', background=False, download=True)
        dset.Omniglot(root=BASEDIR + '/data', background=True, download=True)

        try:
            os.mkdir(processed_dir)
        except OSError:
            pass

        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(origin_dir, p)):
                shutil.move(os.path.join(origin_dir, p, f), processed_dir)

        shutil.rmtree(origin_dir)

    def make_dataset(self, mode):
        x = []
        y = []

        with open(os.path.join(self.vinyals_dir, mode + '.txt'), 'r') as f:
            classes = f.read().splitlines()

        class_to_idx = {string: i for i, string in enumerate(classes)}

        for idx, c in enumerate(tqdm(classes, desc="Making dataset")):
            class_dir, degree = c.rsplit('/', 1)
            degree = int(degree[3:])

            transform = A.Compose([
                A.Resize(28, 28),
                A.Rotate((degree, degree), p=1),
                A.Normalize(mean=0.92206, std=0.08426),
                ToTensorV2(),
            ])

            for img_dir in glob(os.path.join(self.root_dir, class_dir, '*')):
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(image=img)['image']

                x.append(img)
                y.append(idx)
        y = torch.LongTensor(y)
        return x, y, class_to_idx
