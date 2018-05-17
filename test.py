import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
from run import run_model, test_model, run_k_fold, train_model
import math

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])
test_input, test_target = bci.load(root='./data', train=False, store_local=True, one_khz=opt['one_khz'])

split = math.floor(train_̇input.size()[0]/ opt['k_fold'])

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
test_dataset = Dataset(opt, test_input, test_target, log, 'test')
log.info('[Data loaded.]')

model = get_model(opt, train_dataset.input_size())

testing_accuracy = run_model(model, train_dataset, test_dataset, opt, log)
