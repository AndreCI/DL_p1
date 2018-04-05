import torch


class Model(torch.nn.Module):
    def __init__(self):
        # TODO: define what we want in a model
        super(Model, self).__init__()

    def _build(self):
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()
    def train(self):
        raise NotImplementedError()
