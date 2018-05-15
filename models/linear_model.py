import torch
from models.model import Model
from torch.autograd import Variable
import numpy as np

class LinearModel(Model):
    '''
    A simple model which uses Linear layers to solve the problem.
    '''
    def __init__(self, opt, layers):
        '''
        Construct the model respectively to the parameters layers
        :param layers: a list of tuples, either the dimensions of the dense layers or the name of the activation functions
        '''
        super(LinearModel, self).__init__(opt)
        self.layers = []
        self._build(layers)
        self.type = 'Linear'

    def _build(self, layers):
        for l in layers:
            if len(l) == 3:
                in_features = l[0]
                out_features = l[1]
                bias = l[2]
                current_dense = torch.nn.Linear(in_features, out_features, bias=bias)
                torch.nn.init.xavier_normal(current_dense.weight.data)
                self.layers.append(current_dense)
                self.add_module('dense',current_dense)
                self._initialize_param(current_dense.weight)
            elif len(l) == 1:
                if l[0] == 'sigmoid':
                    current_activation = torch.nn.Sigmoid()
                    self.layers.append(current_activation)
                    self.add_module('activation',current_activation)
                elif l[0] == 'tanh':
                    current_activation = torch.nn.Tanh()
                    self.layers.append(current_activation)
                    self.add_module('activation', current_activation)
                elif l[0] == 'relu':
                    current_activation = torch.nn.ReLU()
                    self.layers.append(current_activation)
                    self.add_module('activation', current_activation)
                elif l[0] == 'softmax':
                    current_activation = torch.nn.Softmax()
                    self.layers.append(current_activation)
                    self.add_module('activation', current_activation)
                else:
                    raise NotImplementedError()
        self.layers = torch.nn.ModuleList(self.layers)
        self.dropout_layer = torch.nn.Dropout(self.opt['dropout'])
        self._build_criterion()
        self._build_optimizer()

    def forward(self, x, train=True):
        for l in self.layers:
            x = l(x)
            if isinstance(l, torch.nn.Linear) and train:
                x = self.dropout_layer(x)
        return x.type(torch.FloatTensor)


    def one_step_run(self, example,target, mode='train'):
        features = Variable(example.view(1, -1))

        prediction = self(features, mode=="train")

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
        losses = []
        predictions = []
        i = 0
        if mode == 'train':
            self.train()
        elif mode == 'test':
            self.eval()
        else:
            raise AttributeError('The mode %s is not recognized. Please use train or test' %mode)

        while dataset.has_next_example():
            i+=1
            input, target = dataset.next_example()
            loss, pred = self.one_step_run(input, target, mode=mode)
            loss = loss.data.numpy()[0]
            max_score, pred_class = (torch.max(pred.data, 1))#.numpy()[0]
            predictions.append(pred_class.numpy()[0])
            losses.append(loss)
            total_loss+= loss
            #if i % 10 == 0:
                #print("total_loss:",total_loss/i)
        return losses, predictions