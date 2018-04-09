import torch
from torch.autograd import Variable
from models.model import Model
import numpy as np


class RecurrentModel(Model):
    '''A rather complex model which uses LSTMs to handle time dependencies.'''
    def __init__(self, hidden_units, criterion = 'MSE', optimizer = 'SGD'):
        super(RecurrentModel, self).__init__()
        self._build(hidden_units)
        self.type='Recurrent'
        if criterion == 'MSE':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.parameters()), lr=0.00000001, momentum=0.9)
        else:
            raise NotImplementedError()

    def _build(self, hidden_units):
        self.input_layer = torch.nn.LSTM(input_size=28, hidden_size=hidden_units, num_layers=1)
        self.decoder = torch.nn.Linear(hidden_units,1, bias=True)

    def forward(self, x):
        x, (h, c) = self.input_layer(x)
        x = x[-1].view(-1)
        x = self.decoder(x)
        return x


    def one_step_run(self, example,target, mode='train'):
        example = Variable(example).t().contiguous()
        features = (example.view(example.size()[0], 2, example.size()[1]))

        prediction = self(features)

        temp = torch.FloatTensor([[target]])
        target = Variable(temp)

        if mode == 'train':
            self.optimizer.zero_grad()
            loss = self.criterion(prediction, target)
            #print("prediction:",prediction)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        elif mode == 'test':
            self.optimizer.zero_grad()
            loss = self.criterion(prediction, target)
            return loss, prediction
        else:
            raise NotImplementedError()

    def run(self, examples, targets, mode='train'):
        total_loss = 0.0
        losses = []
        predictions = []
        for i, e in enumerate(examples):
            target = targets[i]
            loss, pred = self.one_step_run(e, target, mode=mode)
            loss = loss.data.numpy()[0]
            pred = pred.data.numpy()[0]
            predictions.append(pred)
            losses.append(loss)
            total_loss+= loss
            if i % 10 == 0:
                print("total_loss:",total_loss/(i+1))
        return total_loss/(i+1), losses, predictions, self.optimizer