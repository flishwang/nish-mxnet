# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
#from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    return fnames

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)

    parser.add_argument('--act-type',type=str,default='',help='if set, replace all the relu in the network')
    parser.add_argument('--nish-ratio', type=float, default=1.0, help='ratio to decide how close nish looks to relu')

    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples  = 50000,
        image_shape    = '3,28,28',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 270,
        lr             = .05,
        lr_step_epochs = '150,200,250',
    )
    args = parser.parse_args()

    if len(args.act_type)>0:
        import symbols.MishOp
        symbols.MishOp.act_type_replace=args.act_type
        symbols.MishOp.nr = args.nish_ratio
    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
