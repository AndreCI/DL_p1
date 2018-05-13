import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
from run import run_model
import time
import math

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])
test_input, test_target = bci.load(root='./data', train=False, store_local=True, one_khz=opt['one_khz'])

split = math.floor(test_input.size()[0] * opt['validation_set_ratio'])
if split != 0:
    validation_input = test_input[:split]
    validation_target = test_target[:split]
    test_input = test_input[split:]
    test_target = test_target[split:]
    validation_dataset = Dataset(opt, validation_input, validation_target, 'val')

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
test_dataset = Dataset(opt, test_input, test_target, log, 'test')

# toy_input, toy_target = generate_toy_data()
# toy_dataset = Dataset(toy_input, toy_target, 'train', remove_DC_level=False, normalize=False)

log.info('[Data loaded.]')

model = get_model(opt, train_dataset)
log.info('[Model build.]')

# epoch_done = model.load_model(log)
epoch_done = 0
run_model(model, train_dataset, test_dataset, opt, log)
# model.save_model(opt['epoch_number'] + epoch_done, log)

exit()


# model.save_model(opt['epoch_number'] + epoch_done, log)

# exit()

#####
# TODO:REMOVE
def seq_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(28 * 50, 20),
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
            max_score, pred_class = (torch.max(result.data, 1))  # .numpy()[0]
            predictions.append(pred_class.numpy()[0])
        while test_dataset.has_next_example():
            ex, target = test_dataset.next_example()
            result = model(torch.autograd.Variable(ex).view(1, -1))
            new_target = torch.autograd.Variable(torch.LongTensor([target]))
            loss = criterion(result, new_target)
            max_score, pred_class = (torch.max(result.data, 1))  # .numpy()[0]
            predictions_test.append(pred_class.numpy()[0])
            avg_loss_test += loss

        acc_train = compute_accuracy(train_dataset, predictions, reduce=False)
        acc_test = compute_accuracy(test_dataset, predictions_test, reduce=False)
        print(str(
            'Train accuracy %i epoch : %f, loss %f' % (i, sum(acc_train) / len(acc_train), avg_loss / len(acc_train))))
        print(str(
            'Test accuracy %i epoch: %f, loss %f' % (i, sum(acc_test) / len(acc_test), avg_loss_test / len(acc_test))))
        print("---")


def seqential():
    for i in range(10):
        log.info('Starting new test epoch.')
        seq = SequentialAutopick()
        t0 = time.time()
        log.info(seq.all_run(train_dataset, test_dataset, 400))
        log.info(time.time() - t0)


seqential()
