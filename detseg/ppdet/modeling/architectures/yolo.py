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

import cv2
import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..post_process import JDEBBoxPostProcess

__all__ = ['YOLOv3']


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 seg_head='MultiSegHead',
                 post_process='BBoxPostProcess',
                 data_format='NCHW',
                 det_loss_weight=0.5,
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            seg_head (nn.Layer): segmentatin head
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.seg_head = seg_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.return_idx = isinstance(post_process, JDEBBoxPostProcess)
        self.det_loss_weight = det_loss_weight

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)
        seg_head = create(cfg['seg_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
            'seg_head': seg_head,
        }

    def _forward(self):
        x_hw = self.inputs['image'].shape[-2:]
        t = cv2.getTickCount()
        body_feats = self.backbone(self.inputs)
        t = cv2.getTickCount() - t
        #print("Backbone %gms" % (t*1000/cv2.getTickFrequency()))
        #for body_feat in body_feats:
        #    print(body_feat.shape)

        t = cv2.getTickCount()
        neck_feats = self.neck(body_feats, self.for_mot)
        t = cv2.getTickCount() - t
        #print("Neck %gms" % (t*1000/cv2.getTickFrequency()))
        #for neck_feat in neck_feats:
        #    print(neck_feat.shape)

        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']

        if self.training:
            # Add seg loss
            seg_loss = self.seg_head(neck_feats, inputs=self.inputs, x_hw=tuple(x_hw))
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            
            # Merge two task losses
            yolo_losses['seg_loss'] = seg_loss
            yolo_losses['yolo_loss'] = yolo_losses['loss']
            yolo_losses['loss'] = (1.-self.det_loss_weight)*seg_loss + self.det_loss_weight*yolo_losses['loss']
            # 'loss': loss, 
            # 'seg_loss': loss_seg,

            # 'yolo_loss': 
            # 'loss_cls': loss_cls, 
            # 'loss_iou': loss_iou,
            # 'loss_dfl': loss_dfl,
            # 'loss_l1': loss_l1,
            

            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses

        else:
            t = cv2.getTickCount()
            yolo_head_outs = self.yolo_head(neck_feats)
            t = cv2.getTickCount() - t
            #print("YOLO head %gms" % (t*1000/cv2.getTickFrequency()))

            t = cv2.getTickCount()
            bbox, bbox_num = self.yolo_head.post_process(
                yolo_head_outs, self.inputs['im_shape'],
                self.inputs['scale_factor'])
            t = cv2.getTickCount() - t
            #print("Det post process %gms" % (t*1000/cv2.getTickFrequency()))

            # Add seg prediction head
            ori_shape = paddle.divide(self.inputs['im_shape'], self.inputs['scale_factor']) 
            logits = self.seg_head(neck_feats, x_hw=(900, 1600))
            
            output = {'bbox': bbox, 'bbox_num': bbox_num, 'logits': logits, 'ori_shape': ori_shape}
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
