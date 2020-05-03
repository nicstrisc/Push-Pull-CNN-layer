from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CorruptionCIFAR(data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


class CIFAR10_C:
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'

    corr_test_folder = 'CIFAR-10-C'
    corr_test_sets = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'gaussian_blur',
        'gaussian_noise',
        'glass_blur',
        'impulse_noise',
        'jpeg_compression',
        'labels',
        'motion_blur',
        'pixelate',
        'saturate',
        'shot_noise',
        'snow',
        'spatter',
        'speckle_noise',
        'zoom_blur'
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    n_cifar = 10000

    def __init__(self, root, transform=None, corr_category='gaussian_noise'):
        self.root = os.path.expanduser(root)
        self.transform = transform

        if not self._check_integrity():
            # download the original CIFAR-10 in case it is not available (needed for GT)
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # Check that the corruption category is correct
        assert (corr_category in self.corr_test_sets)

        file = os.path.join(self.root, self.corr_test_folder) + '/labels.npy'
        self.test_labels = np.load(file)
        self.test_labels = np.int64(self.test_labels)

        # load test data
        file = os.path.join(self.root, self.corr_test_folder) + '/' + corr_category + '.npy'
        self.test_data = np.load(file)

    def get_severity_set(self, set_id):
        imgdata = self.test_data[(set_id - 1) * self.n_cifar:set_id * self.n_cifar]
        return CorruptionCIFAR(imgdata, self.test_labels, transform=self.transform)

    def _check_integrity(self):
        root = self.root
        for fentry in self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100_C(CIFAR10_C):
    corr_test_folder = 'CIFAR-100-C'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
