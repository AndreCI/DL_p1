
import torch
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
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
rec = RecurrentModel(hidden_units=50, dropout=0.7, optimizer='SGD')
convo = ConvolutionalModel()

def convol_model():
    for i in range(20):
        train_dataset.setup_epoch(single_pass=True)
        losses_train, preds_tr = convo.run(train_dataset, mode='train')
        test_dataset.setup_epoch(single_pass=True)
        losses_test, preds_te = convo.run(test_dataset, mode='test')

        annoncement = str("Train loss: %f, test loss: %f" % (
        sum(losses_train) / len(losses_train), sum(losses_test) / len(losses_test)))
        print(annoncement)
        acc_train = compute_accuracy(train_dataset, preds_tr, reduce=False)
        acc_test = compute_accuracy(test_dataset, preds_te, reduce=False)
        print(str(
            'Train accuracy: %f, test accuracy %f' % (sum(acc_train) / len(acc_train), sum(acc_test) / len(acc_test))))
        print("====")
    display_losses(losses_train, losses_test, linear.type, opt, running_mean_param=50)
    display_accuracy(acc_train, acc_test, linear.type, opt, running_mean_param=20)

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
    for i in range(20):
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
        print(i)
        seq = SequentialAutopick()
        print(seq.all_run(train_dataset, test_dataset, 10))

    #print(L)
#rec_model()
convol_model()
#seq_model()
#linear_model()