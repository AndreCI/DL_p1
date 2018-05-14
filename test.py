import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
from run import run_model, test_model
import time
import math
import numpy as np

opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])
test_input, test_target = bci.load(root='./data', train=False, store_local=True, one_khz=opt['one_khz'])

split = math.floor(test_input.size()[0] * opt['validation_set_ratio'])
if split != 0:
    test_input, test_target, validation_input, validation_target = split_test_set(test_input, test_target, split)
    validation_dataset = Dataset(opt, validation_input, validation_target, 'val')

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
test_dataset = Dataset(opt, test_input, test_target, log, 'test')

# toy_input, toy_target = generate_toy_data()
# toy_dataset = Dataset(toy_input, toy_target, 'train', remove_DC_level=False, normalize=False)

log.info('[Data loaded.]')

dropout = np.arange(0, 0.6, 0.1)
hidden_units = [5, 20]
inits = ['xavier_uniform', 'xavier_normal']
depths = [0, 1]
optimizers = [(['Adadelta', 1.0])]
weight_decays = np.arange(0, 0.3, 0.1)

low_passes = np.arange(0, 100, 20)
high_passes = np.arange(0, 10, 5)
normalize = [True, False]
last_ms = [0, 200, 100]
best_acc = 0
best_mod = None
models_saved = []
rhos = [0.9, 0.8, 0.95]
epss = [1e-6, 1e-5]
for rho in rhos:
    opt['rho'] = rho
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
                                opt['exp_name'] = str('CONVO_' + str(d) + '_' + str(h) + '_' + str(i) + '_' + str(dep) + '_' + str(o) + '_' + str(w))
                                model = get_model(opt, train_dataset)
                                log.info('*' * 60)
                                log.info('new model with parameters:')
                                log.info(opt)
                                accuracy = run_model(model, train_dataset, validation_dataset, opt, log)
                                if accuracy >= best_acc:
                                    best_acc = accuracy
                                    best_mod = model
                                    models_saved.append(model)
                                if accuracy > 0.8:
                                    model.save_model(log, str('model_' + opt['exp_name']))
                                log.info('best model: %.3f' %best_acc)
# model.save_model(opt['epoch_number'] + epoch_done, log)
i = 0
for model in models_saved:
    i+=1
    acc = test_model(model, test_dataset, opt, log)
    print(acc)
    if acc > 0.8:
        model.save_model(log, 'valtestedmodel_%i_%.3f' %(i,acc))
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
