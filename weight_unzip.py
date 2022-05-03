import argparse
import json
import os
from utils import find_model
from os.path import join
from cafegan import CAFEGAN

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', type=str, default=None, required=True)
    parser.add_argument('--load_epoch', dest='load_epoch', type=int, default='latest')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--data_save_root', dest='data_save_root', type=str, default=None)
    return parser.parse_args(args)

args_ = parse()
with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.gpu = args_.gpu
args.experiment_name = args_.experiment_name
args.load_epoch = args_.load_epoch
args.betas = (args.beta1, args.beta2)

model = CAFEGAN(args)
model.load(find_model(join(args_.data_save_root, args_.experiment_name, 'checkpoint'), epoch=args_.load_epoch))
model.saveG_D(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights_unzip.{:d}.pth'.format(args.load_epoch)
            ), flag='unzip')