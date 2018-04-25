
import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
import logging
import sys
import time

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)
log.info('[Arg used:]' + str(opt))
train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True)
test_input, test_target = bci.load(root='./data', train=False, store_local=True)
toy_input, toy_target = generate_toy_data()

train_dataset = Dataset(train_̇input, train_̇target, 'train', opt['remove_DC_level'], opt['normalize_data'])
test_dataset = Dataset(test_input, test_target, 'test', opt['remove_DC_level'], opt['normalize_data'])
#toy_dataset = Dataset(toy_input, toy_target, 'train', remove_DC_level=False, normalize=False)

print(train_dataset.inputs.size())
exit()
log.info('[Data loaded.]')

model = get_model(opt)
log.info('[Model build.]')


def run_model(model):
    epoch_number = opt['epoch_number']
    final_loss_train = []
    final_loss_test = []
    final_acc_train = []
    final_acc_test = []
    t0 = time.time()
    for i in range(epoch_number):
        ts = time.time()
        train_dataset.setup_epoch(single_pass=True)
        losses_train, preds_tr = model.run(train_dataset, mode='train')
        test_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = model.run(test_dataset, mode='test')

        annoncement = str("Train loss: %f, test loss: %f" % (
            sum(losses_train) / len(losses_train), sum(losses_test) / len(losses_test)))
        log.info(annoncement)
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        log.info(str(
            'Train accuracy: %f, test accuracy %f' % (sum(acc_train) / len(acc_train), sum(acc_test) / len(acc_test))))
        final_loss_train.append(sum(losses_train) / len(losses_train))
        final_loss_test.append(sum(losses_test) / len(losses_test))
        final_acc_train.append(sum(acc_train) / len(acc_train))
        final_acc_test.append(sum(acc_test) / len(acc_test))
        te = time.time()
        log.info('[Epoch %i/%i done in %.2f s. Approximatly %.2fs. remaining.]' %(i, epoch_number, (te-ts), ((te-ts) * (epoch_number - i))))
    log.info('[Producing accuracy and loss figures.]')
    display_losses(final_loss_train, final_loss_test, model.type, opt, running_mean_param=int(epoch_number/10))
    display_accuracy(final_acc_train, final_acc_test, model.type, opt, running_mean_param=int(epoch_number/10))
    log.info('[Finished in %.2fs.]' %(time.time() - t0))

#epoch_done = model.load_model(log)
epoch_done = 0
run_model(model)
model.save_model(opt['epoch_number'] + epoch_done, log)


#####
#TODO:REMOVE
def seq_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(28*50, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2),
        torch.nn.Softmax()
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(list(model.parameters()))
    for i in range(0, 1000):
        train_dataset.setup_epoch()
        test_dataset.setup_epoch()
        avg_loss = 0.0
        avg_loss_test = 0.0
        predictions = []
        predictions_test = []
        while train_dataset.has_next_example():
            optimizer.zero_grad()
            ex, target = train_dataset.next_example()
            result = model(torch.autograd.Variable(ex).view(1, -1))
            new_target = torch.autograd.Variable(torch.LongTensor([target]))
            loss = criterion(result, new_target)
            avg_loss += loss
            loss.backward()
            optimizer.step()
            max_score, pred_class = (torch.max(result.data, 1))#.numpy()[0]
            predictions.append(pred_class.numpy()[0])
        while test_dataset.has_next_example():
            ex, target = test_dataset.next_example()
            result = model(torch.autograd.Variable(ex).view(1, -1))
            new_target = torch.autograd.Variable(torch.LongTensor([target]))
            loss = criterion(result, new_target)
            max_score, pred_class = (torch.max(result.data, 1))  # .numpy()[0]
            predictions_test.append(pred_class.numpy()[0])
            avg_loss_test+=loss

        acc_train = compute_accuracy(train_dataset, predictions, reduce=False)
        acc_test = compute_accuracy(test_dataset, predictions_test, reduce=False)
        print(str('Train accuracy %i epoch : %f, loss %f' %(i, sum(acc_train)/len(acc_train), avg_loss/len(acc_train))))
        print(str('Test accuracy %i epoch: %f, loss %f' %(i, sum(acc_test)/len(acc_test), avg_loss_test/len(acc_test))))
        print("---")

def seqential():
    for i in range(10):
        log.info('Starting new test epoch.')
        seq = SequentialAutopick()
        t0 = time.time()
        log.info(seq.all_run(train_dataset, test_dataset, 10))
        log.info(time.time() - t0)

seqential()