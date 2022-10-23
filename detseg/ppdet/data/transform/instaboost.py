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

import cv2
import numpy as np
from .operators import register_op, BaseOperator, Resize
from .instaboostfast import InstaBoostConfig, InstaBoostInstances

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'InstaBoost'
]

@register_op
class InstaBoost(BaseOperator):
    r"""Data augmentation method in `InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting
    <https://arxiv.org/abs/1908.07801>`_.

    Refer to https://github.com/GothicAi/Instaboost for implementation details.
    """

    def __init__(self,
                 anno_file='data/train.json',
                 image_dir='data/images',
                 seg_dir='data/',
                 categories=[],
                 config_params=[]):
        #try:
        #    import instaboostfast
        #except ImportError:
        #    raise ImportError(
        #        'Please run "pip install instaboostfast" '
        #        'to install instaboostfast first for instaboost augmentation.')
        super(InstaBoost, self).__init__()

        config_dict = dict()
        for category, config_param in zip(categories, config_params):
            config_dict[category] = InstaBoostConfig().get_dict()
            for k, v in config_param.items():
                config_dict[category][k] = v 

        self.ibi = InstaBoostInstances(anno_file, image_dir, seg_dir, config_dict)

    def _load_anns(self, results):
        labels = results['gt_class'].flatten()
        bboxes = results['gt_bbox']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': [[x1,y1, x2,y1, x2,y2, x1,y2]],
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        #gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            #ann['segmentation'] = [ann['segmentation'][0][:8]]
            #gt_masks_ann.append(np.array(ann['segmentation'], dtype=np.float32))
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1,4)
        gt_labels = np.array(gt_labels, dtype=np.int64).reshape(-1,1)
        #gt_masks_ann = np.array(gt_masks_ann, dtype=np.int64).reshape(-1,8)
        results['gt_class'] = gt_labels.astype(np.int32)
        results['gt_bbox'] = gt_bboxes.astype(np.float32)
        #results['gt_poly'] = gt_masks_ann #.astype(np.float32)
        #print(results['image'].shape, results['image'].dtype)
        #print('new ', img.shape, img.dtype)
        results['image'] = img
        return results

    def __call__(self, results, context=None):
        #t = cv2.getTickCount()
        img = results['image']
        orig_type = img.dtype
        anns = self._load_anns(results)

        # try:
        #     import instaboostfast as instaboost
        # except ImportError:
        #     raise ImportError('Please run "pip install instaboostfast" '
        #                       'to install instaboostfast first.')
        #st = cv2.getTickCount()
        anns, img = self.ibi.get_sync_data_only_obj(anns, img.astype(np.uint8), results)
        #anns, img = self.ibi.get_trans_data(anns, img.astype(np.uint8))

        results = self._parse_anns(results, anns, np.array(img).astype(orig_type))
        #t = cv2.getTickCount() - t
        #print("Instaboost time: %gms" % (t * 1000/cv2.getTickFrequency()))
        return results
