import dlc_bci as bci
import argparse
from util.configuration import get_args, get_model, setup_log
from util.data_util import *
import re
import csv
import json
from collections import OrderedDict
import numpy as np

model = 'BiConvolutinal'
save_dir = 'stored_models'
path = os.path.join(save_dir, model)
path = os.path.join('models', path)
scan = re.compile(r"(paramModel_)(0.[\d]+)(.*)")
first_row = False
to_write = []
with open(os.path.join(path, str(model) + '.csv'), 'w', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for file in os.listdir(path):
        found = (re.match(scan, file))
        if found:
            accuracy = found.group(2)
            with open(os.path.join(path, file), "r", encoding='utf-8') as f:
                data = OrderedDict(sorted(json.load(f).items()))
                data['score'] = accuracy
                if not first_row:
                    writer.writerow(data.keys())
                    first_row = True
                to_write.append([x for x in (data.values())])
                #writer.writerow(data.values())
    print(to_write)
    to_write = sorted(to_write, key= lambda x: x[-1])
    for row in to_write:
        writer.writerow(row)

exit()
opt = get_args(argparse.ArgumentParser())
log = setup_log(opt)

train_̇input, train_̇target = bci.load(root='./data', train=True, store_local=True, one_khz=opt['one_khz'])

train_dataset = Dataset(opt, train_̇input, train_̇target, log, 'train')
train_dataset.display_data(3)