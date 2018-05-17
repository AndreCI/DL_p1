import models.model
import torch
from torch.autograd import Variable
from collections import OrderedDict

class BiConvolutionalModel(models.model.Model):
    def __init__(self, opt, input_shape):
        super(BiConvolutionalModel, self).__init__(opt)
        self._build(input_shape)
        self.type = 'BiConvolutinal'

    def _build(self, input_shape):
        self.input_layer_1 = torch.nn.Conv1d(in_channels=input_shape[0],
                                           out_channels=self.opt['hidden_units'],
                                           kernel_size=input_shape[1])
        self.add_module('input1', self.input_layer_1)
        self.input_layer_2 = torch.nn.Conv1d(in_channels=input_shape[1],
                                             out_channels=self.opt['hidden_units'],
                                             kernel_size=input_shape[0])
        self.add_module('input2', self.input_layer_2)

        if self.opt["activation_type"] == "ELU":
            self.activation_convo = torch.nn.ELU()  # ELU? Or something else? ELU seems good
        elif self.opt["activation_type"] == "RELU":
            self.activation_convo = torch.nn.ReLU()
        elif self.opt["activation_type"] == "sigmoid":
            self.activation_convo = torch.nn.Sigmoid()
        self.activation_hidden = torch.nn.Tanh()
        self.add_module('acti_convo', self.activation_convo)

        self.dropout = torch.nn.Dropout(self.opt['dropout'])
        self.add_module('dropout', self.dropout)
        linears = []
        for i in range(self.opt['depth']):
            new_layer = torch.nn.Linear(self.opt['hidden_units'] * 2, self.opt['hidden_units'])
            name = str('linear_%i' %i)
            new_activation = self.activation_hidden
            linears.append((name, new_layer))
            name = str('activ_%i' %i)
            linears.append((name, new_activation))
            new_dropout_layer = torch.nn.Dropout(self.opt['dropout'])
            name = str('dropout_%i' %i)
            linears.append((name, new_dropout_layer))
            self._initialize_param(new_layer.weight)
            #self._initialize_param(new_layer.bias)
        self.hidden_layers = torch.nn.Sequential(OrderedDict(linears))
        if self.opt['depth'] == 0:
            self.decoder = torch.nn.Linear(self.opt['hidden_units'] * 2, 2)
        else:
            self.decoder = torch.nn.Linear(self.opt['hidden_units'], 2)
        self.add_module('decoder', self.decoder)

        self.output_acti = torch.nn.Softmax()
        self.add_module('Softmax', self.output_acti)

        self._build_criterion()
        self._build_optimizer()
        self._initialize_param(self.input_layer_1.weight)
        self._initialize_param(self.input_layer_2.weight)
        #self._initialize_param(self.input_layer.bias)
        self._initialize_param(self.decoder.weight)
        #self._initialize_param(self.decoder.bias)

    def forward(self, x, train=True):
        x1 = self.input_layer_1(x)
        x2 = self.input_layer_2(torch.transpose(x, 1, 2))
        x1 = x1.view(-1)
        x2 = x2.view(-1)
        x = torch.cat([x1, x2], dim=0)
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

    def reset(self):
        self.input_layer_1.reset_parameters()
        self._initialize_param(self.input_layer_1.weight)
        self.input_layer_2.reset_parameters()
        self._initialize_param(self.input_layer_2.weight)
        for m in self.hidden_layers.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
                self._initialize_param(m.weight)
        self.decoder.reset_parameters()
        self._initialize_param(self.decoder.weight)