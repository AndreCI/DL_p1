import argparse
import os
from models.linear_model import LinearModel
from models.recurrent_model import RecurrentModel
from models.sequential_autopick import SequentialAutopick
from models.convolutional_model import ConvolutionalModel
from models.attentional_recurrent_model import AttentionalRecurrentModel
import logging
import sys

def get_args(parser):
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    #Directory arguments
    parser.add_argument('--save_dir', help="directory to save trained models.", default=root_dir + "/models/stored_models/", type=str)
    parser.add_argument('--fig_dir', help="directory to save different figures.", default=root_dir + "/figs/", type=str)
    parser.add_argument('--log_root', help="directory to save the logging journal", default=root_dir + "/logs/", type=str)
    parser.add_argument('--exp_name', help="name of the experience and the log file. This will not overwrite previous logs with the same name.",
                        default='testval', type=str)

    #General arguments
    parser.add_argument('--model', help="Type of model to use.", default='Sequential', type=str)
    parser.add_argument('--validation_set_ratio', help="fraction of the test set to be converted into a validation set.", default=0.0, type=float)

    #Model arguments
    parser.add_argument('--epoch_number', help="Number of epoch to train.", default=1000, type=int)
    parser.add_argument('--weight_decay', help="Value for the L2 penalty. Set to 0 to not use it.", default=0.0, type=float)
    parser.add_argument('--dropout', help="Probability of the dropout during training. Set to 0 to not use it.", default=0.2, type=float)
    parser.add_argument('--lr', help="Learning rate to train the models.", default=1e-3, type=float)
    parser.add_argument('--optimizer', help="Optimizer used to train the model.", default='Adadelta', type=str)
    parser.add_argument('--momentum', help="Momentum used for the SGD optimizer", default=0.9, type=float)
    parser.add_argument('--criterion', help="Criterion used to evaluate the model.", default='CrossEntropy', type=str)
    parser.add_argument('--hidden_units', help="Number of hidden units used in the model.", default=20, type=int)
    parser.add_argument('--init_type', help="The type of initalization applied to the hidden states and cells (recurrent model).", default="uniform", type=str)
    parser.add_argument('--depth', help="Depth of the different models, if applicable.", default=0, type=int)

    #Data pre processing arguments
    parser.add_argument('--one_khz',
                        help=" If True, creates dataset from the 1000Hz data instead of the default 100Hz.", default=False, type=bool)
    parser.add_argument('--remove_DC_level',
                        help="Remove the DC bias over each channel in all the datasets.", default=True, type=bool)
    parser.add_argument('--low_pass',
                        help="Value of the low pass filter, if any. Set to None to disable.", default=None, type=float)
    parser.add_argument('--high_pass',
                        help="Value of the low pass filter, if any. Set to None to disable.", default=None, type=float)
    parser.add_argument('--normalize_data',
                        help="Normalize data in order to have all channels values between -1 and 1.", default=True, type=bool)
    parser.add_argument('--last_ms', help="Use only the last X miliseconds as features. Set to 0 to disable", default=100, type=int)
    parser.add_argument('--pca_features',
                        help="Replace input by its principal components. Set to 0 to disable.", default=0, type=int)
    parser.add_argument('--cannalwise_pca_features',
                        help="Replace input by the principal components of each cannal. Set to 0 to disable.", default=3, type=int)

    return vars(parser.parse_args())

def get_model(opt, dataset):
    if opt['model']=='Linear':
        input_shape = 1
        for l in dataset.input_size():
            input_shape *= l
        layers = [(input_shape, opt['hidden_units'], True), ['relu'],
                  (opt['hidden_units'], opt['hidden_units'], True), ['softmax'],
                  (opt['hidden_units'], 2, True), ['softmax']]
        linear = LinearModel(opt, layers)
        return linear
    elif opt['model']=='Recurrent':
        rec = RecurrentModel(opt, dataset.input_size())
        return rec
    elif opt['model']=='AttentionRecurrent':
        arec = AttentionalRecurrentModel(opt, dataset.input_size())
        return arec
    elif opt['model']=='Convolutional':
        convo = ConvolutionalModel(opt, dataset.input_size())
        return convo
    elif opt['model']=='Sequential':
        return None
    else:
        raise NotImplementedError('This model has not been yet implemented.')

def setup_log(opt):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    if not os.path.exists(opt['log_root']): os.mkdir(opt['log_root'])
    log_dir = os.path.join(opt['log_root'], opt['model'])
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, opt['exp_name'] + str('_output.log'))
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    log.info('[Program starts.]')
    important_infos = []
    for key in opt:
        if type(opt[key]) is bool:
            if opt[key]:
                important_infos.append({key: opt[key]})
        elif type(opt[key]) is int:
            if opt[key] > 0:
                important_infos.append({key: opt[key]})
        elif type(opt[key]) is str and 'dir' not in key:
            important_infos.append({key: opt[key]})
    log.info('[Arg used:]' + str(important_infos))
    return log

