#!/usr/bin/env python
# -*- coding:utf-8 -*-
# datetime:2022/4/22 15:51


import argparse
import AttentionSVM as at


parser = argparse.ArgumentParser(description="The AttentionSVM method")
parser.add_argument("--embedding", "-e", help="Embedding layer sizes", default="128")
parser.add_argument("--filters1", "-f1", help="first convolutional layer filters", default="512")
parser.add_argument("--filters2", "-f2", help="second convolutional layer filters", default="512")
parser.add_argument("--kernel_size1", "-k1", help="first convolutional layer kernel_sizes", default="3")
parser.add_argument("--kernel_size2", "-k2", help="second convolutional layer kernel_sizes", default="5")
parser.add_argument("--c1", "-c1", help="", required=True)
parser.add_argument("--beta", "-beta", help="", required=True)
parser.add_argument("--input", "-i", help="input folder", required=True)
parser.add_argument("--cross_validation", "-cv", help="K value of cross validation", default="-1")
args = parser.parse_args()

if __name__ == '__main__':
    try:
        at.run(args.embedding, args.filters1, args.filters2, args.kernel_size1, args.kernel_size2, args.c1, args.beta, args.input, args.cross_validation)
    except Exception as e:
        print(e)
