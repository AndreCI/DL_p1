import torch
import os

class Model(torch.nn.Module):
    def __init__(self):
        # TODO: define what we want in a model
        super(Model, self).__init__()
        self.type = 'AbstractModelClass'

    def _build(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def one_step_run(self, *args):
        raise NotImplementedError()

    def run(self, *args):
        raise NotImplementedError()

    def save_model(self, optimizer, epoch, iterInd, opt):
        #TODO:rewrite
        save_dict = dict(
            {'model': self.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'iter': iterInd})
        print('\n')
        print('-' * 60)
        save_name = 'savedModel_E%d_%d.pth' % (epoch, iterInd)
        location = os.path.join(opt.save_dir, self.type)
        if not os.path.exists(location): os.mkdir(location)
        save_name = os.path.join(location, save_name)
        print('Saving Model to : ', location)
        torch.save(save_dict, save_name)
        print('-' * 60)

    @staticmethod
    def load_model(type, opt):
        #TODO: should this be here?
        location = os.path.join(opt.save_dir, type)
        if not os.path.exists(location): raise FileNotFoundError('location not found (%s)' %(location))
        list_files = os.listdir(location)

        print(list_files)
        print(location)
        raise NotImplementedError()
