import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math
import random
from sklearn import decomposition
from scipy.signal import lfilter, butter

class Dataset(object):
    '''A simple class to implement useful methods on data'''
    def __init__(self, opt, inputs, targets, log, type='train'):
        if type != 'train' and type != 'test' and type != 'val':
            raise AttributeError('A dataset should have the train, test or val type.')
        self.inputs = inputs
        self.targets = targets
        self.type = type
        if len(inputs) != len(targets):
            raise AttributeError('A dataset should have the same number of targets and examples.')
        self.length = len(inputs)
        self.single_pass = True
        self.counter = -1
        self.pcas = None

        self.opt=opt
        self.log=log

        if opt['low_pass'] is not None:
            self.apply_low_pass(opt['low_pass'])
        if opt['high_pass'] is not None:
            self.apply_high_pass(opt['high_pass'])
        if opt['remove_DC_level']:
            self.apply_DC_level()
        if opt['normalize_data']:
            self.apply_normalization()
        if opt['last_ms'] > 0:
            self.switch_last_X_miliseconds(opt['last_ms'])
        if opt['pca_features'] > 0:
            self.switch_PCA(opt['pca_features'])
        if opt['cannalwise_pca_features'] > 0:
            self.switch_PCA_cannals(opt['cannalwise_pca_features'])

    def apply_low_pass(self, low_pass_value):
        fs = 1000 if self.opt['one_khz'] else 100
        cutoff = low_pass_value
        for j in range(self.inputs.size()[0]):
            for i in range(self.inputs.size()[1]):
                signal = self.inputs[j, :, i]
                B, A = butter(1, cutoff / (fs / 2), btype='low')  # 1st order Butterworth low-pass
                filtered_signal = lfilter(B, A, signal, axis=0)
                self.inputs[j,:,i] = torch.FloatTensor(filtered_signal)

    def apply_high_pass(self, high_pass_value):
        fs = 1000 if self.opt['one_khz'] else 100
        cutoff = high_pass_value
        for j in range(self.inputs.size()[0]):
            for i in range(self.inputs.size()[1]):
                signal = self.inputs[j, :, i]
                B, A = butter(1, cutoff / (fs / 2), btype='high')  # 1st order Butterworth low-pass
                filtered_signal = lfilter(B, A, signal, axis=0)
                self.inputs[j,:,i] = torch.FloatTensor(filtered_signal)

    def switch_PCA(self, k=3):
        X = self.inputs.view(self.inputs.size()[0], -1)
        X_mean = X.mean(1)
        X = X - X_mean.view(X.size()[0], -1)
        U, _, _ = torch.svd(X.t())
        self.pcas = X.mm(U[:,:k])
        self.pcas = self.pcas.view(self.pcas.size()[0], self.pcas.size()[1], -1)
        self.inputs = self.pcas.view(self.inputs.size()[0], -1)

    def switch_PCA_cannals(self, k=3):
        self.inputs = self.inputs.contiguous()
        extracted_features = torch.zeros((self.inputs.size()[0], self.inputs.size()[1], k))
        for i in range(28):
            pca = decomposition.PCA(n_components=k)
            current_data = self.inputs[:, i, :].contiguous()
            current_feature = (torch.FloatTensor(pca.fit_transform(current_data)).contiguous()).view(self.inputs.size()[0], -1, k)
            if i == 0:
                extracted_features = current_feature
            else:
                extracted_features = torch.cat([extracted_features, current_feature], dim=1)
        self.inputs = extracted_features

    def apply_normalization(self):
        channels_maxs, _ = torch.abs(self.inputs).max(2)
        channels_maxs = channels_maxs.view(self.inputs.size()[0], self.inputs.size()[1], -1)
        self.inputs /= channels_maxs

    def apply_DC_level(self):
        channels_means = self.inputs.mean(2).view(self.inputs.size()[0], self.inputs.size()[1], -1)
        self.inputs -= channels_means

    def switch_last_X_miliseconds(self, ms_to_keep):
        self.inputs = self.inputs[:, :, (50-math.floor(ms_to_keep/10)):]

    def input_size(self):
        return self.inputs[0].size()

    def _shuffle(self):
        permutation = torch.randperm(self.length)
        self.inputs = self.inputs[permutation]
        self.targets = self.targets[permutation]

    def setup_epoch(self, single_pass=True, shuffle=True):
        self.single_pass = single_pass
        if single_pass:
            self.counter = 0
        else:
            self.counter = -1
        if shuffle:
            self._shuffle()

    def has_next_example(self):
        if not self.single_pass:
            return True
        else:
            return self.counter < self.length


    def next_example(self):
        if not self.single_pass:
            x = random.randint(0, self.length - 1)
            return self.inputs[x].clone(), self.targets[x]
        else:
            if self.counter == -1:
                raise Warning('You should call setup_epoch() before calling next_example()')
                self.setup_epoch()
            if self.has_next_example():
                (input, target) = self.inputs[self.counter].clone(), self.targets[self.counter]
                self.counter += 1
                return input, target
            else:
                raise AttributeError("No more example in this dataset.")

    def get_targets(self, mode='binary'):
        raise NotImplementedError()
        #TODO: fix this
        if mode == 'binary':
            return self.targets
        elif mode == 'multiclass':
            target = np.zeros((self.length, 2))
            target[:, 0] = self.targets
            target[:, 1] = 1 - self.targets
            return target
        else:
            raise AttributeError('mode must be binary or multiclass')



def compute_accuracy(dataset, predictions, reduce=True):
    targets = dataset.targets
    targets = targets.numpy()
    results = targets == predictions
    if reduce:
        return sum(results)
    else:
        return (results)
    #print(score/len(results))

def display_losses(train_loss, test_loss, model_type, opt, running_mean_param=1):
    if running_mean_param > len(train_loss) or running_mean_param > len(test_loss):
        running_mean_param = 1
    train_loss = running_mean(train_loss, N=running_mean_param)
    test_loss = running_mean(test_loss, N=running_mean_param)
    plt.figure()
    title = str('Evolution of train and test loss on model %s' %model_type)
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Testing loss'])
    name = str('%s_loss_%s.png' % (model_type, opt['exp_name']))
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)

def display_accuracy(train_accuracy, test_accuracy, model_type, opt, running_mean_param=1):
    if running_mean_param > len(train_accuracy) or running_mean_param > len(test_accuracy):
        running_mean_param = 1
    train_accuracy = running_mean(train_accuracy, N=running_mean_param)
    test_accuracy = running_mean(test_accuracy, N=running_mean_param)
    plt.figure()
    title = str('Evolution of train and test accuracy on model %s' % model_type)
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.legend(['Training accuracy', 'Testing accuracy'])
    name = str('%s_accuracy_%s.png' % (model_type, opt['exp_name']))
    loc = os.path.join(opt['fig_dir'], name)
    plt.savefig(loc)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def generate_toy_data(points_number=1000):
    examples = np.random.uniform(0, 1, (points_number, 2))
    target = np.zeros(points_number)
    for i,ex in enumerate(examples):
        dist = ex[0] * ex[1]
        if dist>0.5:
            target[i] = 1
        else:
            target[i] = 0
    return torch.from_numpy(examples).type(torch.FloatTensor), torch.from_numpy(target).type(torch.LongTensor)