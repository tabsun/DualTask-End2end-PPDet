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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import os
import cv2
import json
import numpy as np
from .operators import register_op, BaseOperator

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'LaneMask'
]

@register_op
class LaneMask(BaseOperator):
    """Add lane mask for input dataset.
    """

    def __init__(self,
                 train_anno_file='dataset/train_det.json',
                 val_anno_file='dataset/val_det.json',
                 mask_dir='dataset/label_mask'):
        super(LaneMask, self).__init__()

        self.mask_dir = mask_dir
        self.id2name = dict()
        data = json.load(open(train_anno_file, 'r'))
        for image in data['images']:
            self.id2name[image['id']] = image['file_name'].replace('.jpg', '.npy')

        data = json.load(open(val_anno_file, 'r'))
        for image in data['images']:
            self.id2name[image['id']] = image['file_name'].replace('.jpg', '.npy')

    def __call__(self, results, context=None):
        image_id = int(results['im_id'])
        mask = np.load(os.path.join(self.mask_dir, self.id2name[image_id]))
        mask = mask.astype(np.int64)
        results['mask'] = mask
        return results
