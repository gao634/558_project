import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

def main(args):
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--path-id', type=int, default=2000)
parser.add_argument('--cmp', default=False, action='store_true')
parser.add_argument('--single-path', default=False, action='store_true')
parser.add_argument('--point-cloud', default=False, action='store_true')
parser.add_argument('--path-file', nargs='*', type=str, default=[], help='path file')
args = parser.parse_args()

print(args)
main(args)