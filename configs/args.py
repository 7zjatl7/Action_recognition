import argparse
import os
import math
import time
import torch


class args_parse():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.init_args()
        
    def init_args(self):
        self.parser.add_argument('--train_dataset', default='train_action_dataset', type=str)
        self.parser.add_argument('--test_dataset', default='test_action_dataset', type=str)
        self.parser.add_argument('--lr', default=1e-4, type=float)
        self.parser.add_argument('--workers', default=8, type=int)
        self.parser.add_argument('--batch_size', default=8, type=int)
        self.parser.add_argument('--nepoch', default=30, type=int)
        self.parser.add_argument('--gpu', default='3', type=str)
        self.parser.add_argument('--train', default=1, type=int)
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--test_epoch', default=3, type=int)
        self.parser.add_argument('--previous_dir', default='', type=str)
        self.parser.add_argument('--previous_best_threshold', type=float, default= -math.inf)
        self.parser.add_argument('--checkpoint', type=str, default='checkpoint/')
        
        
        
    def parse(self):
        return self.parser.parse_args()