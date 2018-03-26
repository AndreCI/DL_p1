
# This is distributed under BSD 3-Clause license

import torch
import numpy
import os
import errno

from six.moves import urllib

def tensor_from_file(root, filename,
                     base_url = 'https://documents.epfl.ch/users/f/fl/fleuret/www/data/bci',
                     store_local=False):
    if store_local:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path, root)
        if not os.path.exists(root): os.mkdir(root)
    file_path = os.path.join(root, filename)
    if not os.path.exists(file_path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = base_url + '/' + filename

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())

    return torch.from_numpy(numpy.loadtxt(file_path))

def load(root, train = True, download = True, one_khz = False, store_local=False):
    """
    Args:

        root (string): Root directory of dataset.

        train (bool, optional): If True, creates dataset from training data.

        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        one_khz (bool, optional): If True, creates dataset from the 1000Hz data instead
            of the default 100Hz.

    """

    nb_electrodes = 28

    if train:

        if one_khz:
            dataset = tensor_from_file(root, 'sp1s_aa_train_1000Hz.txt', store_local=store_local)
        else:
            dataset = tensor_from_file(root, 'sp1s_aa_train.txt', store_local=store_local)

        input = dataset.narrow(1, 1, dataset.size(1) - 1)
        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = dataset.narrow(1, 0, 1).clone().view(-1).long()

    else:
        if one_khz:
            input = tensor_from_file(root, 'sp1s_aa_test_1000Hz.txt', store_local=store_local)
        else:
            input = tensor_from_file(root, 'sp1s_aa_test.txt', store_local=store_local)
        target = tensor_from_file(root, 'labels_data_set_iv.txt', store_local=store_local)

        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = target.view(-1).long()

    return input, target
