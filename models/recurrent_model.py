import torch
from torch.autograd import Variable
from models.model import Model
import numpy as np


class RecurrentModel(Model):
    '''A rather complex model which uses LSTMs to handle time dependencies.'''
    def __init__(self, opt, hidden_units):
        super(RecurrentModel, self).__init__(opt)
        self.hidden_units = hidden_units
        self._build()
        self.type='Recurrent'


    def _build(self):
        self.input_layer = torch.nn.LSTM(input_size=28, hidden_size=self.hidden_units, num_layers=1)
        self.add_module('input', self.input_layer)
        self.dropout_layer = torch.nn.Dropout(self.opt['dropout'])
        self.add_module('dropout', self.dropout_layer)
        self.decoder = torch.nn.Linear(self.hidden_units, 2, bias=True)
        self.add_module('decoder', self.decoder)
        self.softmax = torch.nn.Softmax()
        self.add_module('activation', self.softmax)

        self._build_criterion()
        self._build_optimizer()

    def forward(self, x, train=True):
        x, (h, c) = self.input_layer(x)
        x = x[-1].view(-1)
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