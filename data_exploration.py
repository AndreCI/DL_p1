import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
train_dataset.display_data(3)