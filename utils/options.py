#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_rounds', type=int, default=20, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--model', type=str, default='CNN', help='model name')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', type=bool, default=True, help='aggregation over all clients')
    # -------------------------------------------------------------
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
    parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (1, 1)])
    parser.add_argument('--batch_size', type=int, default=128, help="test batch size")
    parser.add_argument('--num_points', type=int, default=128, help='resampled points number')

    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--early_stop', default=50)

    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr_single', default=0.001, type=float, help="learning rate")  # FTL 1e-4  #single 1e-3

    parser.add_argument('--fine_tune_ep', default=5, type=int, help='fine-tune epochs')
    parser.add_argument('--fine_tune_lr', default=0.0004, type=float, help='learning rate')
    parser.add_argument('--fine_tune_data_ratio', default=1.0, type=float, help='fine-tune ratio')
    parser.add_argument('--save_dir', type=str, default=r'C:\Users\30685\Desktop\FL-SOH\save_FTL\Scenario 2', help='save directory')
    args = parser.parse_args()
    return args
