
import torch
from models.linear_model import LinearModel
import dlc_bci as bci

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True)

layers = [(28*50, 1, True), 'tanh']

linear = LinearModel(layers)

for i in range(1):
    L = linear.run(train_̇input, train_̇target)
    print(L)