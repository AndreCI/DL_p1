import models.model
import torch
from torch.autograd import Variable

class ConvolutionalModel(models.model.Model):
    def __init__(self, criterion = 'CrossEntropy', optimizer='Adagrad'):
        super(ConvolutionalModel, self).__init__()
        self._build()
        if criterion == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion == 'MSE':
            raise NotImplementedError()
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        if optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(list(self.parameters()),lr=1e-3)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.parameters()), lr=1e-3, momentum=0.9)
        else:
            raise NotImplementedError()

    def _build(self):
        self.input_layer = torch.nn.Conv1d(in_channels=28,
                                           out_channels=28,
                                           kernel_size=50)
        self.add_module('input', self.input_layer)
        self.activation_convo = torch.nn.Sigmoid()
        self.add_module('acti_convo', self.activation_convo)
        self.decoder = torch.nn.Linear(28, 2)
        self.add_module('decoder', self.decoder)
        self.output_acti = torch.nn.Softmax()
        self.add_module('Softmax', self.output_acti)


    def forward(self, x, train=True):
        x = self.input_layer(x)
        x = x.view(-1)
        x = self.activation_convo(x)
        x = self.decoder(x)
        x = self.output_acti(x)
        return x.view(1, -1)

    def one_step_run(self, example,target, mode='train'):
        #example = Variable(example).t().contiguous()
        features = Variable(example.view(1, example.size()[0], example.size()[1]))
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