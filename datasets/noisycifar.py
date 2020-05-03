from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
# from utils import noise_models


class NCIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with noisy test set.
    --- Modified to generate noise-augmented test sets

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        noise_test (array, optional): If present, it indicates that the test set has to be
            augmented with noise. Options are:
            - type: gaussian, speckle, poisson
            - val: the std for the gaussian and speckle noise (not used for poisson)

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 noise_test=None, clip_noise=False,
                 normalize_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_transform = normalize_transform

        self.train = train  # training set or test set
        self.noise_test = noise_test
        # if self.noise_test is not None and self.noise_test['type'] == 'poisson':
        #     self.noise_test['val'] = 0
        self.noise_test_data = None
        self.clip_noise = clip_noise

        if not self._check_integrity():
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]

            # load test data
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))

            # Load the noise data (to be used on to the original test data - depending on the kind of noise)
            # the test data are noised by means of a custom Transform class
            if self.noise_test is not None:
                filetest = f + '_gauss_' + str(self.noise_test['val'])
                file = os.path.join(self.root, self.base_folder, filetest)

                if not os.path.exists(file):  # generate the noise samples
                    if self.noise_test['type'] == 'gaussian' or self.noise_test['type'] == 'speckle':
                        self.noise_test_data = np.random.normal(0, self.noise_test['val'] ** 0.5, self.test_data.shape)
                        torch.save(self.noise_test_data, file)
                else:
                    self.noise_test_data = torch.load(file)

            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if not self.train and self.noise_test is not None:  # generate noisy data
            noisemap = None
            if self.noise_test_data is not None:
                noisemap = self.noise_test_data[index]

            '''
            t = NoiseTransform(mode=self.noise_test['type'],
                               value=self.noise_test['val'],
                               noisemap=noisemap)
            img = t(img)
            '''

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.normalize_transform is not None:
            img = self.normalize_transform(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
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

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class NCIFAR100(NCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

'''
class NoiseTransform(object):
    def __init__(self, mode=None, value=None, noisemap=None, clip=False):
        self.mode = mode
        self.val = value
        self.noisemap = noisemap
        self.clip_noise = clip

    def __call__(self, pic):
        noise_pic = pic

        if self.mode == 'gaussian' or self.mode == 'speckle':
            if self.noisemap is not None:
                noise = torch.Tensor(self.noisemap)  # .transpose(self.noisemap, (2, 0, 1)))
                if self.mode == 'gaussian':
                    noise_pic = pic.add(noise)
                elif self.mode == 'speckle':
                    noise_pic = torch.from_numpy(noise_pic.numpy() + noise_pic.numpy() * noise.numpy())
            else:
                noise_pic = torch.from_numpy(noise_models.add_noise(pic.numpy(),
                                                                    mode=self.mode,
                                                                    clip=self.clip_noise,
                                                                    var=self.val))
        elif self.mode == 'poisson':
            pic2 = (pic.numpy() - 0.5) * self.val + 0.5
            pic2 = np.float64(pic2)
            noise_pic = torch.from_numpy(noise_models.add_noise(pic2,
                                                                mode=self.mode,
                                                                clip=self.clip_noise))
            noise_pic = noise_pic.type(torch.FloatTensor)
        elif self.mode == 'contrast':
            pic2 = np.float64((pic.numpy() - 0.5) * self.val + 0.5)
            noise_pic = torch.from_numpy(pic2).type(torch.FloatTensor)
        #
        return noise_pic

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        if self.value is not None:
            format_string += 'var={0}'.format(self.value)
        format_string += ')'
        return format_string
'''