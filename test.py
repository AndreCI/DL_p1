
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
toy_input, toy_target = generate_toy_data()

train_dataset = Dataset(train_̇input, train_̇target, 'train')
test_dataset = Dataset(test_input, test_target, 'test')
toy_dataset = Dataset(toy_input, toy_target, 'train')

layers = [(28 * 50, 20, True), ['sigmoid'], (20, 20, True), ['sigmoid'], (20, 2, True), ['softmax']]

linear = LinearModel(layers, optimizer='Adagrad')
rec = RecurrentModel(hidden_units=30)

def linear_model(): #TODO: remove. Used for debug and exploration phase only
    for i in range(100):
        train_dataset.setup_epoch(single_pass=True)
        losses_train, preds_tr = linear.run(train_dataset, mode='train')
        test_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = linear.run(test_dataset, mode='test')

        annoncement = str("Train loss: %f, test loss: %f" %(sum(losses_train)/len(losses_train), sum(losses_test)/len(losses_test)))
        print(annoncement)
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        print(str('Train accuracy: %f, test accuracy %f' %(sum(acc_train)/len(acc_train), sum(acc_test)/len(acc_test))))
        print("====")
    display_losses(losses_train, losses_test, linear.type, opt, running_mean_param=50)
    display_accuracy(acc_train, acc_test, linear.type, opt, running_mean_param=20)

def rec_model(): #TODO: remove. Used for debug and exploration phase only
    for i in range(10):
        train_dataset.setup_epoch()
        losses_train, preds_tr = rec.run(train_dataset, mode='train')
        test_dataset.setup_epoch()
        losses_test, preds_te = rec.run(test_dataset, mode='test')


        annoncement = str("Train loss: %f, test loss: %f" %(sum(losses_train)/len(losses_train), sum(losses_test)/len(losses_test)))
        print(annoncement)
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        print(str('Train accuracy: %f, test accuracy %f' %(sum(acc_train)/len(acc_train), sum(acc_test)/len(acc_test))))
        print("====")
    display_losses(losses_train, losses_test, rec.type, opt, running_mean_param=50)
    display_accuracy(acc_train, acc_test, rec.type, opt, running_mean_param=20)

    #print(L)
rec_model()
#linear_model()