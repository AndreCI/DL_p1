import torch
from models.model import Model
from torch.autograd import Variable

class LinearModel(Model):
    '''
    A simple model which uses Linear layers to solve the problem.
    '''
    def __init__(self, layers):
        '''
        Construct the model respectively to the parameters layers
        :param layers: a list of tuples, either the dimensions of the dense layers or the name of the activation functions
        '''
        super(LinearModel, self).__init__()
        self.layers = []
        for l in layers:
            if len(l) == 3:
                in_features = l[0]
                out_features = l[1]
                bias = l[2]
                current_dense = torch.nn.Linear(in_features, out_features, bias=bias)
                self.layers.append(current_dense)
                self.add_module('dense',current_dense)
            elif len(l) == 4:
                if l == 'tanh':
                    current_activation = torch.nn.Sigmoid()
                    self.layers.append(current_activation)
                    self.add_module('activation',current_activation)
                else:
                    raise NotImplementedError()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return (x) + 0.0001


    def one_step_run(self, example,target, criterion, optimizer, mode='train'):
        features = Variable(example.view(1, -1))

        prediction = self(features)

        temp = torch.FloatTensor([[target]])
        target = Variable(temp)

        if mode == 'train':
            optimizer.zero_grad()
            loss = criterion(prediction, target)
            #print("prediction:",prediction)
            loss.backward()
            optimizer.step()
            return loss, prediction
        else:
            raise NotImplementedError()



    def run(self, examples, targets):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(list(self.parameters()), lr=0.00000001, momentum=0.9)
        total_loss = 0.0
        for i, e in enumerate(examples):
            target = targets[i]
            loss, pred = self.one_step_run(e, target, criterion, optimizer)
            total_loss+= loss
            #print("loss:",loss)
            if i % 10 == 0:
                print("total_loss:",total_loss/(i+1))
        return total_loss/(i+1)