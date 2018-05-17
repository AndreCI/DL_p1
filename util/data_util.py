import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import math
import random
from sklearn import decomposition
from scipy.signal import lfilter, butter, iirnotch

class Dataset(object):
    '''A class to implement useful methods on data'''
    def __init__(self, opt, inputs, targets, log, type='train'):
        '''
        Initialiaze a datase
        :param opt: the options, used to know which transformation to apply to the data
        :param inputs: the inputs data
        :param targets: the targets
        :param log: the log, used to display information
        :param type: the type of the dataset, which ca be either train, test or val
        '''
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
        if opt['notch_filter'] is not None:
            self.apply_notch_filter(opt['notch_filter'])
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

    def apply_notch_filter(self, f0):
        '''
        Apply a notch filter to the data, filtering the given frequency
        :param f0: the frequency to elimilate
        '''
        fs = 1000 if self.opt['one_khz'] else 100
        Q = 30
        w0 = f0/(fs/2)
        for j in range(self.inputs.size()[0]):
            for i in range(self.inputs.size()[1]):
                signal = self.inputs[j, :, i]
                B, A = iirnotch(w0, Q)
                filtered_signal = lfilter(B, A, signal, axis=0)
                self.inputs[j, : , i] = torch.FloatTensor(filtered_signal)

    def apply_low_pass(self, low_pass_value):
        '''
        Apply a low pass filter
        :param low_pass_value: the upper bound
        '''
        fs = 1000 if self.opt['one_khz'] else 100
        cutoff = low_pass_value
        for j in range(self.inputs.size()[0]):
            for i in range(self.inputs.size()[1]):
                signal = self.inputs[j, :, i]
                B, A = butter(1, cutoff / (fs / 2), btype='low')  # 1st order Butterworth low-pass
                filtered_signal = lfilter(B, A, signal, axis=0)
                self.inputs[j,:,i] = torch.FloatTensor(filtered_signal)

    def apply_high_pass(self, high_pass_value):
        '''
        Apply a high pass filter
        :param high_pass_value: the lower bound
        '''
        fs = 1000 if self.opt['one_khz'] else 100
        cutoff = high_pass_value
        for j in range(self.inputs.size()[0]):
            for i in range(self.inputs.size()[1]):
                signal = self.inputs[j, :, i]
                B, A = butter(1, cutoff / (fs / 2), btype='high')  # 1st order Butterworth low-pass
                filtered_signal = lfilter(B, A, signal, axis=0)
                self.inputs[j,:,i] = torch.FloatTensor(filtered_signal)

    def switch_PCA(self, k=3):
        '''
        Uses PCA on the data and transform them into k different components for each example
        :param k: the number of componenent to keep
        '''
        X = self.inputs.view(self.inputs.size()[0], -1)
        X_mean = X.mean(1)
        X = X - X_mean.view(X.size()[0], -1)
        U, _, _ = torch.svd(X.t())
        self.pcas = X.mm(U[:,:k])
        self.pcas = self.pcas.view(self.pcas.size()[0], self.pcas.size()[1], -1)
        self.inputs = self.pcas.view(self.inputs.size()[0], -1)

    def switch_PCA_cannals(self, k=3):
        '''
        Apply PCA to each channel, reducing the data timelength into principal componenet
        :param k: the number of componenent to keep for each channel
        '''
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
        '''
        Normalize the data, i.e. set them between -1 and 1
        '''
        channels_maxs, _ = torch.abs(self.inputs).max(2)
        channels_maxs = channels_maxs.view(self.inputs.size()[0], self.inputs.size()[1], -1)
        self.inputs /= channels_maxs

    def apply_DC_level(self):
        '''
        Remove the DC level to each example, i.e. the mean of each channel.
        '''
        channels_means = self.inputs.mean(2).view(self.inputs.size()[0], self.inputs.size()[1], -1)
        self.inputs -= channels_means

    def switch_last_X_miliseconds(self, ms_to_keep):
        '''
        Only keep the last X milisecondes of the data, reducing the input space
        :param ms_to_keep: the number of milisecondes to keep.
        '''
        self.inputs = self.inputs[:, :, (50-math.floor(ms_to_keep/10)):]

    def input_size(self):
        return self.inputs[0].size()

    def _shuffle(self):
        '''
        Shuffle the data (inputs and targets)
        '''
        permutation = torch.randperm(self.length)
        self.inputs = self.inputs[permutation]
        self.targets = self.targets[permutation]

    def setup_epoch(self, single_pass=True, shuffle=True):
        '''
        Setup the dataset for the next epoch
        :param single_pass: If single pass is true, the dataset will only be looked once
        :param shuffle: If shuffle is true, the dataset will be shuffle.
        '''
        self.single_pass = single_pass
        if single_pass:
            self.counter = 0
        else:
            self.counter = -1
        if shuffle:
            self._shuffle()

    def has_next_example(self):
        '''
        Utility function to know whether or not the dataset can provide a new example
        :return: a boolean
        '''
        if not self.single_pass:
            return True
        else:
            return self.counter < self.length


    def next_example(self):
        '''
        Return a new example and its target, if possible
        :return: a couple (input, target)
        '''
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

    def display_data(self, example_number):
        '''
        Produce a figure representing an example and its target from two views. Mainly used for data exploration
        :param example_number: the example number to look at.
        '''
        example = self.inputs[example_number]
        target = self.targets[example_number]
        if not os.path.exists(self.opt['fig_dir']): os.mkdir(self.opt['fig_dir'])
        save_dir = os.path.join(self.opt['fig_dir'], 'examples')
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        example = example.numpy()
        fig = plt.figure(figsize=(40, 20))
        s = str("Display of example %i. Target is %i" % (example_number, int(target)))
        plt.title(s)
        ax = fig.add_subplot(121, projection="3d")
        x_axes = np.arange(np.shape(example)[0])
        y_axes = np.arange(np.shape(example)[1])
        X, Y = np.meshgrid(y_axes, x_axes)
        ax.plot_surface(X, Y, example, cmap=cm.coolwarm)
        ax.set_xlabel('time')
        ax.set_ylabel('channel')
        ax = fig.add_subplot(122, projection="3d")
        x_axes = np.arange(np.shape(example)[1])
        y_axes = np.arange(np.shape(example)[0])
        X, Y = np.meshgrid(y_axes, x_axes)
        ax.plot_surface(X, Y, np.transpose(example), cmap=cm.coolwarm)
        ax.set_ylabel('time')
        ax.set_xlabel('channel')
        s = str('%i_test_%i.png' % (int(target), example_number))
        path = os.path.join(save_dir, s)
        plt.savefig(fname=path)
        fig.clf()



def compute_accuracy(dataset, predictions, reduce=True):
    '''
    Compute the accuracy of a model based on the dataset and its prediction
    :param dataset: the dataset
    :param predictions: the predictions of the model for the given dataset
    :param reduce: if True, this will return the number of correct guess. If False, this will return a list of boolean indicating which guess was right
    :return:
    '''
    targets = dataset.targets
    targets = targets.numpy()
    results = targets == predictions
    if reduce:
        return sum(results)
    else:
        return (results)


def display_losses(train_loss, test_loss, model_type, opt, running_mean_param=1):
    '''
    Produces a figure showing the evolution of train and test loss.
    :param train_loss: an array containg the training losses
    :param test_loss:  an array containg the testing losses
    :param model_type: the type of the model, which can be accessed by using model.type
    :param opt: the option, used to know where to save the figures, etc.
    :param running_mean_param: the parameter used to smooth the data.
    '''
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
    '''
    Produces a figure showing the evolution of train and test accuracy.
    :param train_loss: an array containg the training accuracies
    :param test_loss:  an array containg the testing accuracies
    :param model_type: the type of the model, which can be accessed by using model.type
    :param opt: the option, used to know where to save the figures, etc.
    :param running_mean_param: the parameter used to smooth the data.
    '''
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
    '''
    Perform a simple running mean
    :param x: the data
    :param N: the running mean parameter
    :return: the data smoothed
    '''
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def split_trainset(train_inputs, train_targets, split, fold_number=0):
    '''
    Split the train set accordingly to the kfold method
    :param train_inputs: the training inputs
    :param train_targets: the training targets
    :param split: the split fraction
    :param fold_number: the fold number
    :return: a tuple containg (train_inputs, train_targets, val_inputs, val_targets)
    '''
    lower_bound = split * fold_number
    upper_bound = split * (fold_number + 1)
    validation_input = train_inputs.clone()[lower_bound: upper_bound]
    validation_target = train_targets.clone()[lower_bound: upper_bound]
    if lower_bound != 0: #Crashes if train.inputs.clone().narrow(0, 0, 0) is used.
        new_train_inputs = train_inputs.clone().narrow(0, 0, lower_bound)
        new_train_targets = train_targets.clone().narrow(0, 0, lower_bound)
        if upper_bound != train_inputs.size()[0]:# Crashes if train.inputs.clone().narrow(0, X, X) is used.
            new_train_inputs = torch.cat([new_train_inputs, train_inputs.clone().narrow(0, upper_bound, train_inputs.size()[0] - upper_bound)])
            new_train_targets = torch.cat([new_train_targets, train_targets.clone().narrow(0, upper_bound, train_targets.size()[0] - upper_bound)])
    else:
        new_train_inputs = train_inputs.clone().narrow(0, upper_bound, train_inputs.size()[0] - upper_bound)
        new_train_targets = train_targets.clone().narrow(0, upper_bound, train_targets.size()[0] - upper_bound)
    return new_train_inputs, new_train_targets, validation_input, validation_target