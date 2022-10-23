# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('eval')


def run(cfg):
    # init parallel environment if nranks > 1
    init_parallel_env()

    # build trainer
    trainer = Trainer(cfg, mode='eval')

    # load weights
    trainer.load_weights(cfg.weights)

    # training
    trainer.evaluate()


def main():
    cfg = load_config('configs/ppyoloe/ppyoloe_crn_m_300e_xd.yml')
    cfg.use_npu = False
    cfg.use_xpu = False
    for k, v in cfg.items():
        print(k, v)
    
    run(cfg)


if __name__ == '__main__':
    main()
