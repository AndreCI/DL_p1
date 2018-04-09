
import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
import dlc_bci as bci
import argparse
from util.configuration import add_arg #TODO:rename
from util.data_util import *

parser = add_arg(argparse.ArgumentParser())
opt = parser.parse_args()


train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True)
test_input, test_target = bci.load(root='./data', train=False, store_local=True)


layers = [(28*50, 2, True),  ['softmax']]

linear = LinearModel(layers, optimizer='Adagrad')
rec = RecurrentModel(hidden_units=3)

for i in range(100):
    total_loss, losses_train, preds_tr, opti = linear.run(train_̇input, train_̇target, mode='train')
    total_test_loss, losses_test, preds_te, opti = linear.run(test_input, test_target, mode='test')
    annoncement = str("Train loss: %f, test loss: %f" %(total_loss, total_test_loss))
    print(annoncement)
    compute_accuracy(train_̇target, preds_tr)
    compute_accuracy(test_target, preds_te)
    print("====")
    #display_losses(losses_train, losses_test, linear.type, opt, running_mean_param=50)

exit()
print('new model')

for i in range(10):
    print('new epoch')
    L,losses_train, optimizer = linear.run(train_̇input, train_̇target, mode='train')
    total_loss, losses_test, opti = linear.run(test_input, test_target, mode='test')

    display_losses(losses_train, losses_test, linear.type, opt, running_mean_param=50)

    #print(L)

