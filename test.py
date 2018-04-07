
import torch
from models.linear_model import LinearModel
import dlc_bci as bci
import argparse
from util.configuration import add_arg #TODO:rename

parser = add_arg(argparse.ArgumentParser())
opt = parser.parse_args()


train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True)

layers = [(28*50, 5, True), 'tanh', (5, 5, True), 'tanh', (5, 5, True), 'tanh',(5, 5, True), 'tanh',(5, 1, True), 'tanh']

linear = LinearModel(layers)

for i in range(100):
    L, optimizer = linear.run(train_̇input, train_̇target)
    #linear.save_model(optimizer, i, 0, opt)
    print(L)