import torch
import os
import re

class Model(torch.nn.Module):
    def __init__(self, opt):
        # TODO: define what we want in a model
        super(Model, self).__init__()
        self.type = 'AbstractModelClass'
        self.opt = opt

    def _build_criterion(self):
        if self.opt['criterion'] == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.opt['criterion'] == 'MSE':
            raise NotImplementedError()
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

    def _build_optimizer(self):
        if self.opt['optimizer'] == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(list(self.parameters()),
                                                 lr=self.opt['lr'],
                                                 weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.parameters()),
                                             lr=self.opt['lr'],
                                             momentum=self.opt['momentum'],
                                             weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(list(self.parameters()),
                                                  lr=self.opt['lr'],
                                                  weight_decay=self.opt['weight_decay'])
        else:
            raise NotImplementedError()

    def _build(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def one_step_run(self, *args):
        raise NotImplementedError()

    def run(self, dataset, mode='train'):
        raise NotImplementedError()

    def save_model(self, epoch, log):
        save_dict = dict(
            {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'criterion': self.criterion.state_dict()})
        log.info('-' * 100)
        save_name = 'savedModel_E%d.pt' % (epoch)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        log.info('[Saving Model at location %s.]' %save_name)
        torch.save(save_dict, save_name)
        log.info('-' * 100)

    def load_model(self, log):
        log.info('-' * 100)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): raise FileNotFoundError('location not found (%s)' %(location))
        list_files = os.listdir(location)
        max_iter=0
        best_model=None
        for l in list_files:
            current_iter = int(re.search(r'\d+', str(l)).group())
            if current_iter>max_iter:
                max_iter=current_iter
                best_model=l
        if max_iter==0:
            raise FileNotFoundError('No model found in location: %s' %location)
        best_file = os.path.join(location,best_model)
        data = (torch.load(best_file))
        log.info('[Loaded model at location %s.]' %best_file)
        self.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.criterion.load_state_dict(data['criterion'])
        log.info('-' * 100)