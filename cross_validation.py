import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
from run import run_model, test_model, run_k_fold, train_model
import time
import math
import numpy as np

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])
test_input, test_target = bci.load(root='./data', train=False, store_local=True, one_khz=opt['one_khz'])

split = math.floor(train_̇input.size()[0]/ opt['k_fold'])

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
test_dataset = Dataset(opt, test_input, test_target, log, 'test')

# toy_input, toy_target = generate_toy_data()
# toy_dataset = Dataset(toy_input, toy_target, 'train', remove_DC_level=False, normalize=False)

log.info('[Data loaded.]')

opt['model'] = 'Recurrent'
dropout = [0.2, 0.5]
hidden_units = [20]
activations = ["RELU", 'sigmoid']
inits = ['xavier_normal']#, 'xavier_uniform']
depths = [0, 1, 2]
optimizers = [(['Adadelta', 1.0])]#, (['Adagrad', 0.015])]
weight_decays = [0, 0.01]
rhos = ["GRU"]

low_passes = np.arange(0, 100, 20)
high_passes = np.arange(0, 10, 5)
normalize = [True, False]
last_ms = [0, 200, 100]
best_acc = 0
best_mod = None
models_saved = []

epss = [1e-6]
models = []

for rho in rhos:
    opt['recurrent_cell_type'] = rho
    for eps in epss:
        opt['eps'] = eps
        for d in dropout:
            opt['dropout'] = d
            for h in hidden_units:
                opt['hidden_units'] = h
                for i in inits:
                    opt['init_type'] = i
                    for dep in depths:
                        opt['depth'] = dep
                        for o in optimizers:
                            opt['optimizer'] = o[0]
                            opt['lr'] = o[1]
                            for w in weight_decays:
                                opt['weight_decay'] = w
                                for act in activations:
                                    opt['activation_type'] = act
                                    model = get_model(opt.copy(), train_dataset.input_size())
                                    models.append(model)

models_scores = run_k_fold(models, train_̇input, train_̇target, opt, log)
best_accuracy = 0
for current in models_scores:
    if current[1] > best_accuracy:
        acc = train_model(current[0], train_dataset, opt, log)
        acc = test_model(current[0], test_dataset, opt, log)


for low in low_passes:
    opt['low_pass'] = low
    for high in high_passes:
        opt['high_pass'] = high
        for ms in last_ms:
            opt['last_ms'] = ms
