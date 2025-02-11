import torch
from torch.autograd import Variable
from models.model import Model
import numpy as np


class AttentionalRecurrentModel(Model):
    '''A rather complex model which uses LSTMs to handle time dependencies.'''
    def __init__(self, opt, input_shape):
        raise NotImplementedError()
        super(AttentionalRecurrentModel, self).__init__(opt)
        self.hidden_units = opt['hidden_units']
        self._build(input_shape)
        self.type='AttentionRecurrent'


    def _build(self, input_shape):
        self.input_layer = torch.nn.LSTM(input_size=input_shape[0], hidden_size=self.hidden_units, num_layers=1)
        self.add_module('input', self.input_layer)

        self.attention = torch.nn.Linear(self.hidden_units, self.hidden_units)
        self.add_module('attention', self.attention)
        self.tanh = torch.nn.Tanh()
        self.vt = Variable(torch.zeros(self.hidden_units).uniform_(0, 1))

        self.dropout_layer = torch.nn.Dropout(self.opt['dropout'])
        self.add_module('dropout', self.dropout_layer)
        self.decoder = torch.nn.Linear(self.hidden_units, 2, bias=True)
        self.add_module('decoder', self.decoder)
        self.softmax = torch.nn.Softmax()
        self.add_module('activation', self.softmax)

        self.hidden_states = Variable(torch.zeros(1, 1, self.hidden_units))
        self.cells_states = Variable(torch.zeros(1, 1, self.hidden_units))
        self._build_criterion()
        self._build_optimizer()

    def _init_state(self):
        if self.opt['init_type'] == "zero":
            return Variable(torch.zeros(1, 1, self.hidden_units))
        elif self.opt['init_type'] == "gaussian":
            return Variable(torch.normal(means=torch.zeros(1, 1, self.hidden_units)))
        elif self.opt['init_type'] == "uniform":
            return Variable(torch.zeros(1, 1, self.hidden_units).uniform_(0, 1))
        else:
            raise NotImplementedError("This init type has not been implemented yet.")

    def forward(self, x, train=True):
        print(x)
        x, (self.h, self.c) = self.input_layer(x, (self._init_state(), self._init_state()))
        print("ere",self.h)
        print("x:",x)
        jac = self.attention(self.h)
        print(jac)

        x = x[-1].view(-1)
        print(x)
        exit()
        if train:
            x = self.dropout_layer(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x.type(torch.FloatTensor).view(-1, 2)


    def one_step_run(self, example,target, mode='train'):
        example = Variable(example).t().contiguous()
        features = (example.view(example.size()[0], 1, example.size()[1]))

        prediction = self(features, mode == 'train')
        #temp = torch.FloatTensor([[target]])
        #target = Variable(temp)
        #print(target)

        v_target = Variable(torch.LongTensor([target]))
        if mode == 'train':
            self.optimizer.zero_grad()
            loss = self.criterion(prediction, v_target)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        elif mode == 'test':
            loss = self.criterion(prediction, v_target)
            return loss, prediction
        else:
            raise NotImplementedError()

    def run(self, dataset, mode='train'):
        total_loss = 0.0
        i = 0
        losses = []
        predictions = []
        while dataset.has_next_example():
            i+=1
            input, target = dataset.next_example()
            loss, pred = self.one_step_run(input, target, mode=mode)
            loss = loss.data.numpy()[0]
            max_score, pred_class = (torch.max(pred.data, 1))  #.numpy()[0]
            predictions.append(pred_class)
            losses.append(loss)
            total_loss+= loss
            #if i % 10 == 0:
                #print("total_loss:",total_loss/i)
        return losses, predictions