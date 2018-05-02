import models.model
import torch
from torch.autograd import Variable
from collections import OrderedDict

class ConvolutionalModel(models.model.Model):
    def __init__(self, opt, input_shape):
        super(ConvolutionalModel, self).__init__(opt)
        self._build(input_shape)
        self.type = 'Convolutinal'

    def _build(self, input_shape):
        self.input_layer = torch.nn.Conv1d(in_channels=input_shape[0],
                                           out_channels=self.opt['hidden_units'],
                                           kernel_size=input_shape[1])
        self.add_module('input', self.input_layer)

        self.activation_convo = torch.nn.Sigmoid()
        self.activation_hidden = torch.nn.Tanh()
        self.add_module('acti_convo', self.activation_convo)

        self.dropout = torch.nn.Dropout(self.opt['dropout'])
        self.add_module('dropout', self.dropout)
        linears = []
        for i in range(self.opt['depth']):
            new_layer = torch.nn.Linear(self.opt['hidden_units'], self.opt['hidden_units'])
            name = str('linear_%i' %i)
            new_activation = self.activation_hidden
            linears.append((name, new_layer))
            name = str('activ_%i' %i)
            linears.append((name, new_activation))
            new_dropout_layer = torch.nn.Dropout(self.opt['dropout'])
            name = str('dropout_%i' %i)
            linears.append((name, new_dropout_layer))
        self.hidden_layers = torch.nn.Sequential(OrderedDict(linears))
        self.decoder = torch.nn.Linear(self.opt['hidden_units'], 2)
        self.add_module('decoder', self.decoder)

        self.output_acti = torch.nn.Softmax()
        self.add_module('Softmax', self.output_acti)

        self._build_criterion()
        self._build_optimizer()

    def forward(self, x, train=True):
        x = self.input_layer(x)
        x = x.view(-1)
        x = self.activation_convo(x)
        if train:
            x = self.dropout(x)
        if self.opt['depth'] != 0:
            x = self.hidden_layers(x)
        x = self.decoder(x)
        x = self.output_acti(x)
        return x.view(1, -1)

    def one_step_run(self, example,target, mode='train'):
        features = Variable(example.view(1, example.size()[0], example.size()[1]))
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
            #if i % 10 == 0:
                #print("total_loss:",total_loss/i)
        return losses, predictions