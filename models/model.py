import torch
import os

class Model(torch.nn.Module):
    def __init__(self):
        # TODO: define what we want in a model
        super(Model, self).__init__()

    def _build(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def one_step_run(self, *args):
        raise NotImplementedError()

    def run(self, *args):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()

    def save_model(self, optimizer, epoch, iterInd, opt):
        #TODO:rewrite
        save_dict = dict(
            {'model': self.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'iter': iterInd})
        print('\n')
        print('-' * 60)
        save_name = 'savedModel_E%d_%d.pth' % (epoch, iterInd)

        save_name = os.path.join(opt.save_dir, save_name)
        print('Saving Model to : ', opt.save_dir)
        torch.save(save_dict, save_name)
        print('-' * 60)

    def load_model(self):
        #TODO: should this be here?
        raise NotImplementedError()
