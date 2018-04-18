import torch
import models.model

class SequentialAutopickModel(models.model.Model):
    def __init__(self):
        super(SequentialAutopickModel, self).__init__()
        self.type = 'sequential_autopick'

    def _build(self):
        pass

    def forward(self, x):
        pass

    def one_step_run(self, *args):
        pass

    def run(self, dataset, mode='train'):
        pass