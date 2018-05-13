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
                                                 lr_decay=self.opt['lr_decay'],
                                                 weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(list(self.parameters()),
                                             lr=self.opt['lr'],
                                             momentum=self.opt['momentum'],
                                             weight_decay=self.opt['weight_decay'])
        elif self.opt['optimizer'] == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(list(self.parameters()),
                                                  lr=self.opt['lr'],
                                                  rho=self.opt['rho'],
                                                  eps=self.opt['eps'],
                                                  weight_decay=self.opt['weight_decay'],
                                                  )
        else:
            raise NotImplementedError()

    def _initialize_param(self, param):
        if self.opt['init_type'] == None:
            pass
        elif self.opt['init_type'] == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform(param)
        elif self.opt['init_type'] == 'kaiming_normal':
            torch.nn.init.kaiming_normal( torch.nn.init.param)
        elif self.opt['init_type'] == 'uniform':
            torch.nn.init.uniform(param)
        elif self.opt['init_type'] == 'xavier_uniform':
            torch.nn.init.xavier_uniform(param)
        elif self.opt['init_type'] == 'xavier_normal':
            torch.nn.init.xavier_normal(param)
        else:
            raise NotImplementedError()

    def _build(self):
        raise NotImplementedError()

    def forward(self, x, train=True):
        raise NotImplementedError()

    def one_step_run(self, *args):
        raise NotImplementedError()

    def run(self, dataset, mode='train'):
        raise NotImplementedError()

    def save_model(self, epoch, log):
        save_dict = dict(
            {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'criterion': self.criterion.state_dict()})
        log.info('-' * 100)
        save_name = 'savedModel_E%s.pt' % (epoch)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        log.info('[Saving Model at location %s.]' %save_name)
        torch.save(save_dict, save_name)
        log.info('-' * 100)

    def load_model(self, log, epoch_number=0):
        log.info('-' * 100)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(self.opt['save_dir']): os.mkdir(self.opt['save_dir'])
        if not os.path.exists(location):
            log.warning('Location not found (%s)' %(location))
            return 0
        list_files = os.listdir(location)
        max_iter=0
        best_model=None
        for l in list_files:
            current_iter = int(re.search(r'\d+', str(l)).group())
            if current_iter>max_iter and epoch_number==0:
                max_iter=current_iter
                best_model=l
            elif current_iter==epoch_number:
                best_model=l
        if max_iter==0:
            log.warning('No model found in location: %s' %location)
            return 0
        best_file = os.path.join(location,best_model)
        data = (torch.load(best_file))
        log.info('[Loaded model at location %s.]' %best_file)
        self.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.criterion.load_state_dict(data['criterion'])
        log.info('-' * 100)
        return max_iter