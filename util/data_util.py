import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import math
import random

class Dataset(object):
    '''A simple class to implement useful methods on data'''
    def __init__(self, inputs, targets, type='train', remove_DC_level=True):
        if type != 'train' and type != 'test' and type != 'dev':
            raise AttributeError('A dataset should have the train, test or dev type.')
        self.inputs = inputs
        self.targets = targets
        self.type = type
        if len(inputs) != len(targets):
            raise AttributeError('A dataset should have the same number of targets and examples.')
        self.length = len(inputs)
        self.single_pass = True
        self.counter = -1

        self._DC_leveld = False
        if remove_DC_level:
            self.switch_DC_level()

    def _setup_DC_level(self):
        self.channels_means = self.inputs.mean(2).view(self.inputs.size()[0], self.inputs.size()[1], -1)

    def switch_DC_level(self):
        if self._DC_leveld:
            self.inputs += self.channels_means
        elif not self._DC_leveld:
            self._setup_DC_level()
            self.inputs -= self.channels_means
        self._DC_leveld = not self._DC_leveld

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
            print(x)
            return self.inputs[x], self.targets[x]
        else:
            if self.counter == -1:
                raise Warning('You should call setup_epoch() before calling next_example()')
                self.setup_epoch()
            if self.has_next_example():
                (input, target) = self.inputs[self.counter], self.targets[self.counter]
                self.counter += 1
                return input, target
            else:
                raise AttributeError("No more example in this dataset.")

    def get_targets(self, mode='binary'):
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
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Testing loss'])
    name = str('%s_loss_evolution.png' % model_type)
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
    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.legend(['Training accuracy', 'Testing accuracy'])
    name = str('%s_accuracy_evolution.png' % model_type)
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