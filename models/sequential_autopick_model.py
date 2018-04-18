import torch
from collections import OrderedDict
import copy
import util.data_util

class SequentialAutopickModel(object):
    def __init__(self):
        super(SequentialAutopickModel, self).__init__()
        self.type = 'sequential_autopick'
        self.all_models = []
        self.scores = []
        self.optimizers = []
        self.activations = [torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Softmax()]
        self.deep = [1, 2, 3]
        self.hidden = [10, 20, 30]
        self._build()
        self.criterion = torch.nn.CrossEntropyLoss()

    def _build(self):
        module_number = 0
        for h in self.hidden:
            for a in self.activations:
                for d in self.deep:
                    current_net = [('first', torch.nn.Linear(28*50, h)), ('first_a', a)]
                    last_current_nets = [copy.deepcopy(current_net)]
                    for i in range(d):
                        temp_current_nets = []
                        for temp in last_current_nets:
                            for a in self.activations:
                                net = copy.deepcopy(temp)
                                act = a
                                module_number+=1
                                hidden_str = str('hidden_%i' %module_number)
                                act_str = str('act_%i' %module_number)
                                net.append((hidden_str, torch.nn.Linear(h, h)))
                                net.append((act_str, act))
                                temp_current_nets.append(net)
                        last_current_nets = copy.deepcopy(temp_current_nets)

                    for temp in last_current_nets:
                        temp.append(('output_layer', torch.nn.Linear(h, 2)))
                        temp.append(('output_activation', torch.nn.Softmax()))
                        current_net = OrderedDict(temp)
                    final_net = torch.nn.Sequential(current_net)
                    self.optimizers.append(torch.optim.Adagrad(list(final_net.parameters())))
                    self.all_models.append(final_net)
        print('Constructed %i models.' %len(self.all_models))

    def all_run(self, train_dataset, val_dataset, epoch_number=10):
        for j in range(epoch_number):
            for i,model in enumerate(self.all_models):
                train_dataset.setup_epoch()
                avg_loss = 0.0
                while train_dataset.has_next_example():
                    self.optimizers[i].zero_grad()
                    ex, target = train_dataset.next_example()
                    result = model(torch.autograd.Variable(ex).view(1, -1))
                    new_target = torch.autograd.Variable(torch.LongTensor([target]))
                    loss = self.criterion(result, new_target)
                    avg_loss += loss
                    loss.backward()
                    self.optimizers[i].step()
                print("model %i/%i went though epoch %i" %(i, len(self.all_models), j))

        for i,model in enumerate(self.all_models):
            val_dataset.setup_epoch()
            avg_loss = 0.0
            predictions = []
            while val_dataset.has_next_example():
                self.optimizers[i].zero_grad()
                ex, target = val_dataset.next_example()
                result = model(torch.autograd.Variable(ex).view(1, -1))
                new_target = torch.autograd.Variable(torch.LongTensor([target]))
                loss = self.criterion(result, new_target)
                avg_loss += loss
                max_score, pred_class = (torch.max(result.data, 1))  # .numpy()[0]
                predictions.append(pred_class.numpy()[0])
                avg_loss += loss
            acc = util.data_util.compute_accuracy(val_dataset, predictions, reduce=True)
            self.scores.append(acc)
        print(self.scores)
        return self.all_models[self.scores.index(max(self.scores))], max(self.scores)