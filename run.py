import time
from util.data_util import compute_accuracy, display_accuracy, display_losses, Dataset, split_trainset
import os
import math

def run_k_fold(models, train_input, train_target, opt, log):
    '''
    Performs cross validation on the list of models, and returns their respective scores.
    :param models: a list of models
    :param train_input: the training inputs
    :param train_target: the training targets
    :param opt: the option
    :param log: the log to display info
    :return: a list containing couple (model, score)
    '''
    split = math.floor(train_input.size()[0] / opt['k_fold'])
    scores = []
    i = 0
    for model in models:
        avg_score = 0.0
        log.info('New model for k_fold (%i k_fold to do). %i/%i models done.' %(opt['k_fold'], i, len(models)))
        log.info('Current option used:')
        log.info(model.opt)
        for k in range(opt['k_fold']):
            current_train_̇input, current_train_̇target, validation_input, validation_target = split_trainset(train_input, train_target, split, fold_number=k)
            validation_dataset = Dataset(opt, validation_input, validation_target, 'val')
            train_dataset = Dataset(opt, current_train_̇input, current_train_̇target, log, 'train')
            score = run_model(model, train_dataset, validation_dataset, model.opt, log)
            model.reset()
            avg_score += score
        i+=1
        avg_score /= opt['k_fold']
        scores.append([model, avg_score])
    return scores

def train_model(model, train_dataset, opt, log):
    '''
    Train a model on the given dataset
    :param model: a model
    :param train_dataset: the training dataset
    :param opt: the options
    :param log: the log
    :return: the best accuracy of the model on the train set
    '''
    epoch_number = opt['epoch_number']
    final_loss_train = []
    final_acc_train = []
    best_train_acc = (0, 0)
    t0 = time.time()
    wait_early = 0
    best_test_loss = 0
    for i in range(epoch_number):
        ts = time.time()
        train_dataset.setup_epoch(single_pass=True)
        losses_train, preds_tr = model.run(train_dataset, mode='train')
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        if opt['verbose'] is 'high':
            annoncement = str("Train loss: %.3f, train accuracy: %.3f" % (
                sum(losses_train) / len(losses_train), sum(acc_train) / len(acc_train)))
            log.info(annoncement)
        final_loss_train.append(sum(losses_train) / len(losses_train))
        final_acc_train.append(sum(acc_train) / len(acc_train))
        end_epoch = i
        train_loss = sum(losses_train) /len(losses_train)
        if train_loss < best_test_loss:
            best_test_loss = train_loss
            wait_early = 1
        elif wait_early > opt['patience']:
            end_epoch = i
            log.info('Early stopping at epoch %i, train loss is %f' % (i, train_loss))
            break
        else:
            wait_early += 1
        if final_acc_train[-1] > best_train_acc[0]:
            best_train_acc = (final_acc_train[-1], i)
        te = time.time()
        if opt['verbose'] is 'high' or (opt['verbose'] is 'medium' and i % 100 == 0):
            log.info('[Epoch %i/%i done in %.3f s. Approximatly %.3fs. remaining.]' % (
                i, epoch_number, (te - ts), ((te - ts) * (epoch_number - i))))
    log.info('First best training accuracy: %.3f at epoch %i' % (best_train_acc[0], best_train_acc[1]))
    log.info('[Finished in %.3fs.]' % (time.time() - t0))
    return best_train_acc[0]

def run_model(model, train_dataset, validation_dataset, opt, log):
    '''
    Train and test a model simultaneously.
    :param model: a model
    :param train_dataset: the train dataset
    :param validation_dataset: the validation dataset
    :param opt: the options
    :param log: the log
    :return: the best accuracy on the testset.
    '''
    epoch_number = opt['epoch_number']
    final_loss_train = []
    final_loss_test = []
    final_acc_train = []
    final_acc_test = []
    best_test_acc = (0, 0)
    t0 = time.time()
    wait_early = 0
    best_test_loss = 0
    for i in range(epoch_number):
        ts = time.time()
        train_dataset.setup_epoch(single_pass=True)
        losses_train, preds_tr = model.run(train_dataset, mode='train')
        validation_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = model.run(validation_dataset, mode='test')
        #Model trained and tested
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(validation_dataset, preds_te, reduce=False)
        final_loss_train.append(sum(losses_train) / len(losses_train))
        final_loss_test.append(sum(losses_test) / len(losses_test))
        final_acc_train.append(sum(acc_train) / len(acc_train))
        final_acc_test.append(sum(acc_test) / len(acc_test))
        test_loss = sum(losses_test) / len(losses_test)
        if opt['verbose'] is 'high':
            annoncement = str("Train loss: %.3f, test loss: %.3f" % (
                final_loss_train[-1], final_loss_test[-1]))
            log.info(annoncement)
            log.info(str(
                'Train accuracy: %.3f, test accuracy %.3f' % (
                final_acc_train[-1], final_acc_test[-1])))
        #Early stopping
        end_epoch = i
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            wait_early = 1
        elif wait_early > opt['patience']:
            end_epoch = i
            log.info('Early stopping at epoch %i, test loss is %f' %(i, test_loss))
            break
        else:
            wait_early +=1
        #Saving best accuracy
        if final_acc_test[-1] > best_test_acc[0]:
            best_test_acc = (final_acc_test[-1], i)
        te = time.time()
        if opt['verbose'] is 'high' or (opt['verbose'] is 'medium' and i%100==0):
            log.info('[Epoch %i/%i done in %.3f s. Approximatly %.3fs. remaining.]' % (
            i, epoch_number, (te - ts), ((te - ts) * (epoch_number - i))))
    log.info('First best validation accuracy: %.3f at epoch %i' % (best_test_acc[0], best_test_acc[1]))
    #Display loss and accuracy
    if opt['verbose'] == 'high' or opt['verbose'] == 'medium' and opt['produce_figures']:
        log.info('[Producing accuracy and loss figures.]')
    if opt['produce_figures']:
        display_losses(final_loss_train, final_loss_test, model.type, opt, running_mean_param=int(end_epoch/ 10))
        display_accuracy(final_acc_train, final_acc_test, model.type, opt, running_mean_param=int(end_epoch / 10))
    log.info('[Finished in %.3fs.]' % (time.time() - t0))
    return best_test_acc[0]


def test_model(model, test_dataset, opt, log):
    '''
    Test a model on the test set. Does not train!
    :param model: a model
    :param test_dataset: the test dataset
    :param opt: the options
    :param log: the log
    :return: the best accuracy on the test set
    '''
    epoch_number = opt['epoch_number']
    final_loss_test = []
    final_acc_test = []
    best_test_acc = (0, 0)
    t0 = time.time()
    for i in range(epoch_number):
        ts = time.time()
        test_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = model.run(test_dataset, mode='test')
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        if opt['verbose'] is 'high':
            annoncement = str("Test loss: %.3f, Test accuracy %.3f" % (
                sum(losses_test) / len(losses_test), sum(acc_test) / len(acc_test)))
            log.info(annoncement)
        final_loss_test.append(sum(losses_test) / len(losses_test))
        final_acc_test.append(sum(acc_test) / len(acc_test))
        if final_acc_test[-1] > best_test_acc[0]:
            best_test_acc = (final_acc_test[-1], i)
        te = time.time()
        if opt['verbose'] is 'high' or (opt['verbose'] is 'medium' and i%100==0):
            log.info('[Epoch %i/%i done in %.3f s. Approximatly %.3fs. remaining.]' % (
            i, epoch_number, (te - ts), ((te - ts) * (epoch_number - i))))
    log.info('First best test accuracy: %.3f at epoch %i' % (best_test_acc[0], best_test_acc[1]))
    log.info("Saving params...")
    id = time.time()
    name = str(str(best_test_acc[0]) + str(id))
    model.save_params(name, log)
    if best_test_acc[0] > 0.65:
        log.info('Saving model...')
        model.save_model(name, log)
    log.info('[Finished in %.3fs.]' % (time.time() - t0))
    return best_test_acc[0]
