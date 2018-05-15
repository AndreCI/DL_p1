import time
from util.data_util import compute_accuracy, display_accuracy, display_losses
import os
def run_model(model, train_dataset, test_dataset, opt, log):
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
        test_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = model.run(test_dataset, mode='test')
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        if opt['verbose'] is 'high':
            annoncement = str("Train loss: %.3f, test loss: %.3f" % (
                sum(losses_train) / len(losses_train), sum(losses_test) / len(losses_test)))
            log.info(annoncement)
            log.info(str(
                'Train accuracy: %.3f, test accuracy %.3f' % (
                sum(acc_train) / len(acc_train), sum(acc_test) / len(acc_test))))
        final_loss_train.append(sum(losses_train) / len(losses_train))
        final_loss_test.append(sum(losses_test) / len(losses_test))
        final_acc_train.append(sum(acc_train) / len(acc_train))
        final_acc_test.append(sum(acc_test) / len(acc_test))
        test_loss = sum(losses_test) / len(losses_test)
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
        if final_acc_test[-1] > best_test_acc[0]:
            best_test_acc = (final_acc_test[-1], i)
        te = time.time()
        if opt['verbose'] is 'high' or (opt['verbose'] is 'medium' and i%100==0):
            log.info('[Epoch %i/%i done in %.3f s. Approximatly %.3fs. remaining.]' % (
            i, epoch_number, (te - ts), ((te - ts) * (epoch_number - i))))
    log.info('First best test accuracy: %.3f at epoch %i' % (best_test_acc[0], best_test_acc[1]))
    if best_test_acc[0] > 0.70:
        model.save_model(str(str(opt['epoch_number']) + str(best_test_acc[0])), log)
    if opt['verbose'] is 'high' or 'medium':
        log.info('[Producing accuracy and loss figures.]')
    #display_losses(final_loss_train, final_loss_test, model.type, opt, running_mean_param=int( end_epoch/ 10))
    #display_accuracy(final_acc_train, final_acc_test, model.type, opt, running_mean_param=int(end_epoch / 10))
    log.info('[Finished in %.3fs.]' % (time.time() - t0))
    return best_test_acc[0]


def test_model(model, test_dataset, opt, log):
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
        #test_loss = sum(losses_test) / len(losses_test)
        if final_acc_test[-1] > best_test_acc[0]:
            best_test_acc = (final_acc_test[-1], i)
        te = time.time()
        if opt['verbose'] is 'high' or (opt['verbose'] is 'medium' and i%100==0):
            log.info('[Epoch %i/%i done in %.3f s. Approximatly %.3fs. remaining.]' % (
            i, epoch_number, (te - ts), ((te - ts) * (epoch_number - i))))
    log.info('First best test accuracy: %.3f at epoch %i' % (best_test_acc[0], best_test_acc[1]))
    if best_test_acc[0] > 0.72:
        id = time.time()
        name = str(str(best_test_acc[0]) + str(id))
        model.save_params(name, log)
        model.save_model(name, log)
    if opt['verbose'] is 'high' or 'medium':
        log.info('[Producing accuracy and loss figures.]')
    #display_losses(final_loss_train, final_loss_test, model.type, opt, running_mean_param=int( end_epoch/ 10))
    #display_accuracy(final_acc_train, final_acc_test, model.type, opt, running_mean_param=int(end_epoch / 10))
    log.info('[Finished in %.3fs.]' % (time.time() - t0))
    return best_test_acc[0]
