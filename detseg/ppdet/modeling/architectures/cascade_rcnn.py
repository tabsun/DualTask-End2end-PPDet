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

import paddle
import cv2
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['CascadeRCNN']


@register
class CascadeRCNN(BaseArch):
    """
    Cascade R-CNN network, see https://arxiv.org/abs/1712.00726

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
        mask_head (object): `MaskHead` instance
        mask_post_process (object): `MaskPostProcess` instance
    """
    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None,
                 mask_head=None,
                 mask_post_process=None):
        super(CascadeRCNN, self).__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
        self.neck = neck
        self.mask_head = mask_head
        self.mask_post_process = mask_post_process
        self.with_mask = mask_head is not None

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)

        out_shape = neck and out_shape or bbox_head.get_head().out_shape
        kwargs = {'input_shape': out_shape}
        mask_head = cfg['mask_head'] and create(cfg['mask_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
            "mask_head": mask_head,
        }

    def _forward(self):
        # inputs : dict
        #          'image': [1, 3, 768, 1344] Paddle Tensor
        #print(self.inputs.keys())
        #print(type(self.inputs['image']))
        #print(self.inputs['image'].shape)

        # body features: list of Paddle Tensor
        #                body feats:
        #                [1, 256, 192, 336] stride=4
        #                [1, 512, 96, 168]  stride=8
        #                [1, 1024, 48, 84]  stride=16
        #                [1, 2048, 24, 42]  stride=32
        t = cv2.getTickCount()
        body_feats = self.backbone(self.inputs)
        t = cv2.getTickCount() - t
        #print("Backbone: %gms" % (t*1000/cv2.getTickFrequency()))

        # body features: list of Paddle Tensor
        #                [1, 256, 192, 336] stride=4
        #                [1, 256, 96, 168]  stride=8
        #                [1, 256, 48, 84]   stride=16
        #                [1, 256, 24, 42]   stride=32
        #                [1, 256, 12, 21]   stride=64
        t = cv2.getTickCount()
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        t = cv2.getTickCount() - t
        #print("Neck: %gms" % (t*1000/cv2.getTickFrequency()))
        #for feat in body_feats:
        #    print(type(feat))
        #    print(feat.shape)

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, bbox_feat = self.bbox_head(body_feats, rois, rois_num,
                                                  self.inputs)
            rois, rois_num = self.bbox_head.get_assigned_rois()
            bbox_targets = self.bbox_head.get_assigned_targets()
            if self.with_mask:
                mask_loss = self.mask_head(body_feats, rois, rois_num,
                                           self.inputs, bbox_targets, bbox_feat)
                return rpn_loss, bbox_loss, mask_loss
            else:
                return rpn_loss, bbox_loss, {}
        else:
            # rois_num: Paddle Tensor only one number: 1000
            # rois:     [1000, 4] Paddle Tensor storing 1000 rois, its number is decided by RNHead/test_proposal/post_nms_top_n in config file.          
            t = cv2.getTickCount()
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            t = cv2.getTickCount() - t
            #print("RPN head: %gms" % (t*1000/cv2.getTickFrequency()))

            # preds: (deltas, scores)
            #     deltas: [1000, 4] Paddle Tensor : The bounding boxes
            #     scores: [1000, 8] Paddle Tensor : The box score on [bg,] + 7 categories
            t = cv2.getTickCount()
            preds, _ = self.bbox_head(body_feats, rois, rois_num, self.inputs)
            t = cv2.getTickCount() - t
            #print("BBox head: %gms" % (t*1000/cv2.getTickFrequency()))
            #print('preds:')
            #print(preds[0].shape, preds[1].shape)

            # refined_rois: list only have one Paddle Tensor
            #               [1000, 4] rois
            t = cv2.getTickCount()
            refined_rois = self.bbox_head.get_refined_rois()
            t = cv2.getTickCount() - t
            #print("Get refined rois: %gms" % (t*1000/cv2.getTickFrequency()))
            #print('refined_rois:')
            #print(refined_rois[0].shape)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']

            # bbox: Paddle Tensor [39, 6] 
            #       each row means category, score, xmin, ymin, xmax, ymax
            # bbox_num Paddle Tensor [39]
            t = cv2.getTickCount()
            bbox, bbox_num = self.bbox_post_process(
                preds, (refined_rois, rois_num), im_shape, scale_factor)


            # bbox: Paddle Tensor [39, 6]
            # bbox_pred: Paddle Tensor [39, 6]
            # bbox_num: Paddle Tensor [1] : 39
            # rescale the prediction back to origin image
            bbox, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            t = cv2.getTickCount() - t
            #print("Post process & get pred: %gms" % (t*1000/cv2.getTickFrequency())) 

            if not self.with_mask:
                return bbox_pred, bbox_num, None
            mask_out = self.mask_head(body_feats, bbox, bbox_num, self.inputs)
            origin_shape = self.bbox_post_process.get_origin_shape()
            mask_pred = self.mask_post_process(mask_out, bbox_pred, bbox_num,
                                               origin_shape)
            return bbox_pred, bbox_num, mask_pred

    def get_loss(self, ):
        rpn_loss, bbox_loss, mask_loss = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(bbox_loss)
        if self.with_mask:
            loss.update(mask_loss)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        output = {
            'bbox': bbox_pred,
            'bbox_num': bbox_num,
        }
        if self.with_mask:
            output.update({'mask': mask_pred})
        return output
