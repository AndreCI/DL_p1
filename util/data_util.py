import numpy as np
import torch
import matplotlib.pyplot as plt
import os

class Dataset(object):
    '''A simple class to implement useful methods on data'''
    def __init__(self, inputs, targets, type='train'):
        if type is not 'train' or type is not 'test' or type is not 'dev':
            raise AttributeError('A dataset should have the train, test or dev type.')
        self.inputs = inputs
        self.targets = targets
        self.type = type
        if len(inputs) != len(targets):
            raise AttributeError('A dataset should have the same number of targets and examples.')
        self.length = len(inputs)

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



def compute_accuracy(targets, predictions):
    targets = targets.numpy()
    results = targets == predictions
    score = sum(results)
    print(score/len(results))

def display_losses(train_loss, test_loss, model_type, opt, running_mean_param=1):
    train_loss = running_mean(train_loss, N=running_mean_param)
    test_loss = running_mean(test_loss, N=running_mean_param)
    plt.figure()
    title = str('Evolution of train and test loss on model %s' %model_type)
    plt.title(title)
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.xlabel('iteration number')
    plt.ylabel('loss')
    name = str('%s_loss_evolution.png' % model_type)
    loc = os.path.join(opt.fig_dir, name)
    plt.savefig(loc)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)