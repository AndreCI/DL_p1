from collections import OrderedDict
import torch
from torch.autograd import Variable
from models.model import Model
import numpy as np


class RecurrentModel(Model):
    '''A rather complex model which uses LSTMs to handle time dependencies.'''
    def __init__(self, opt, input_shape):
        super(RecurrentModel, self).__init__(opt)
        self.hidden_units = opt['hidden_units']
        self._build(input_shape)
        self.type='Recurrent'


    def _build(self, input_shape):
        if self.opt['recurrent_cell_type'] =='LSTM':
            self.input_layer = torch.nn.LSTM(input_size=input_shape[0],
                                             hidden_size=self.hidden_units,
                                             num_layers=1,
                                             dropout=self.opt['dropout'])
        elif self.opt['recurrent_cell_type'] =='GRU':
            self.input_layer = torch.nn.GRU(input_size=input_shape[0],
                                            hidden_size=self.hidden_units,
                                            num_layers=1,
                                            dropout=self.opt['dropout'])
        else:
            raise AttributeError('This cell type (%s) is not recognized. Please use LSTM or GRU.' %self.opt['recurrent_cell_type'])
        self.add_module('input', self.input_layer)
        self.activation_hidden = torch.nn.Sigmoid()
        linears = []
        for i in range(self.opt['depth']):
            new_layer = torch.nn.Linear(self.opt['hidden_units'], self.opt['hidden_units'])
            name = str('linear_%i' % i)
            new_activation = self.activation_hidden
            linears.append((name, new_layer))
            name = str('activ_%i' % i)
            linears.append((name, new_activation))
            new_dropout_layer = torch.nn.Dropout(self.opt['dropout'])
            name = str('dropout_%i' % i)
            linears.append((name, new_dropout_layer))
            self._initialize_param(new_layer.weight)
        self.hidden_layers = torch.nn.Sequential(OrderedDict(linears))

        self.decoder = torch.nn.Linear(self.hidden_units, 2, bias=True)
        self.add_module('decoder', self.decoder)
        self.softmax = torch.nn.Softmax()
        self.add_module('activation', self.softmax)


        #self.hidden_states = Variable(torch.zeros(1, 1, self.hidden_units))
        #self.cells_states = Variable(torch.zeros(1, 1, self.hidden_units))
        self._build_criterion()
        self._build_optimizer()
        for w in self.input_layer.all_weights[0]:
            if w.dim() >= 2:
                self._initialize_param(w)
        self._initialize_param(self.decoder.weight)

    def _init_state(self):
        if self.opt['init_type'] == "zero":
            return Variable(torch.zeros(1, 1, self.hidden_units))
        elif self.opt['init_type'] == "gaussian":
            return Variable(torch.normal(means=torch.zeros(1, 1, self.hidden_units)))
        elif self.opt['init_type'] == "uniform":
            return Variable(torch.zeros(1, 1, self.hidden_units).uniform_(0, 1))
        elif self.opt['init_type'] == 'xavier_uniform' or self.opt['init_type'] == 'xavier_normal':
            v = Variable(torch.zeros(1, 1, self.hidden_units))
            self._initialize_param(v)
        else:
            raise NotImplementedError("This init type (%s) has not been implemented yet." %self.opt['init_type'])

    def forward(self, x, train=True):
        if self.opt['recurrent_cell_type'] == 'LSTM':
            x, (self.h, self.c) = self.input_layer(x, (self._init_state(), self._init_state()))
        elif self.opt['recurrent_cell_type'] == 'GRU':
            x, self.h = self.input_layer(x, self._init_state())
        x = x[-1].view(-1)
        if self.opt['depth'] != 0:
            x = self.hidden_layers(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x.type(torch.FloatTensor).view(-1, 2)


    def one_step_run(self, example,target, mode='train'):
        example = Variable(example).t().contiguous()
        features = (example.view(example.size()[0], 1, example.size()[1]))

        prediction = self(features, mode == 'train')

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
        if mode == 'train':
            self.train()
        elif mode == 'test':
            self.eval()
        else:
            raise AttributeError('The mode %s is not recognized. Please use train or test' %mode)
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
        return losses, predictions