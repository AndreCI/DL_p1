import argparse
import os

def add_arg(parser):
    #TODO: really useful?
    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--save-dir', dest="save_dir", help="directory to save trained models.", default=root_dir + "/models/stored_models/", type=str)
    parser.add_argument('--fig-dir', dest="fig_dir", help="directory to save different figures.", default=root_dir + "/figs/", type=str)
    return parser