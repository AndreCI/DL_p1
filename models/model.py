import torch
import os
import re
import json

class Model(torch.nn.Module):
    '''
    An abstract class to represent a model
    '''
    def __init__(self, opt):
        super(Model, self).__init__()
        self.type = 'AbstractModelClass' #Should be overwritten in the init of each model
        self.opt = opt #save the options

    def _build_criterion(self):
        '''
        Build a criterion for the model
        '''
        if self.opt['criterion'] == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.opt['criterion'] == 'MSE':
            raise NotImplementedError('Criterion MSE not implemented yet.')
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError('Criterion %s not implemented yet.' %self.opt['criterion'])

    def _build_optimizer(self):
        '''
        Build an optimizer for the model
        '''
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
            raise NotImplementedError('The optimizer %s has not been implemented yet' % self.opt['optimizer'])

    def _initialize_param(self, param):
        '''
        Initialize the parameters of a model.
        :param param: The parameter to initialize
        '''
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
            raise NotImplementedError('Initialization %s not implemented yet' %self.opt['init_type'])

    def _build(self):
        '''
        Build the model. Each model must overrides this.
        '''
        raise NotImplementedError()

    def forward(self, x, train=True):
        '''
        Compute the forward pass. Each model must overrides this.
        :param x: the current example
        :param train: whether the mode is training or testing
        '''
        raise NotImplementedError()

    def one_step_run(self, *args):
        '''
        Perform a one step run. Each model must overrides this.
        '''
        raise NotImplementedError()

    def run(self, dataset, mode='train'):
        '''
        Performs a run on the given dataset. Each model must overrides this.
        :param dataset: the dataset
        :param mode: whether the model is training or testing
        '''
        raise NotImplementedError()

    def reset(self):
        '''
        Reset the model. Each model must overrides this
        '''
        raise NotImplementedError()

    def save_model(self, name, log):
        '''
        Save the model in a .pt file.
        :param name: the name of the file
        :param log: the log, to display info
        '''
        save_dict = dict(
            {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'criterion': self.criterion.state_dict()})
        log.info('-' * 100)
        save_name = 'savedModel_%s.pt' % (name)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        if self.opt['verbose'] == "high":
            log.info('[Saving Model at location %s.]' %save_name)
            log.info('-' * 100)
        torch.save(save_dict, save_name)

    def save_params(self, name, log):
        '''
        Save the model parameters in a .json file
        :param name: the name of the file
        :param log: the log
        '''
        data = self.opt
        save_name = 'paramModel_%s.json' % name
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        with open(save_name, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        if self.opt['verbose'] == "high":
            log.info('[Saving Model at location %s.]' %save_name)
            log.info('-' * 100)

    def load_params(self, name, log):
        '''
        Load the params from a given name
        :param name: the name of the file
        :param log: the log
        :return: the data, if any has been found.
        '''
        save_name = 'paramModel_%s.json' % name
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        with open(save_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.opt['verbose'] == "high":
            log.info('[Loading Model at location %s.]' % save_name)
            log.info('-' * 100)
        return data

    def load_model_from_id(self, name, log):
        '''
        Load a model from a given name.
        :param name: the name of the model to load
        :param log: the log
        :return: True if the model was successfully loaded, else False
        '''
        log.info('-' * 100)
        location = os.path.join(self.opt['save_dir'], self.type)
        if not os.path.exists(self.opt['save_dir']): os.mkdir(self.opt['save_dir'])
        if not os.path.exists(location):
            log.warning('Location not found (%s)' % (location))
            return 0
        file=None
        list_files = os.listdir(location)
        for l in list_files:
            if name in l:
                file = l
        if file is None:
            log.warning('No model found in location: %s' % location)
            return False
        file = os.path.join(location, file)
        data = torch.load(file)
        if self.opt['verbose'] == "high":
            log.info('[Loaded model at location %s.]' % file)
        self.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.criterion.load_state_dict(data['criterion'])
        log.info('-' * 100)
        return True

    def load_model_from_epoch(self, log, epoch_number=0):
        '''
        Old method to load a model depending on the number of epoch. This method will load the model which received the
        most training if epoch_number is set to 0. Sort of deprecated.
        :param log: the log
        :param epoch_number: the epoch number to load
        :return: the number of epochs that the model trained
        '''
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
                max_iter = epoch_number
        if max_iter==0:
            log.warning('No model found in location: %s' %location)
            return 0
        best_file = os.path.join(location,best_model)
        data = (torch.load(best_file))
        if self.opt['verbose'] == "high":
            log.info('[Loaded model at location %s.]' %best_file)
        self.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.criterion.load_state_dict(data['criterion'])
        log.info('-' * 100)
        return max_iter