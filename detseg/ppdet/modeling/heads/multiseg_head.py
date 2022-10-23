# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ppdet.core.workspace import register
from ppdet.modeling.losses import OhemCrossEntropyLoss

__all__ = ['MultiSegHead']

class Activation(nn.Layer):
    """
    The wrapper of activations.

    Args:
        act (str, optional): The activation name in lowercase. It must be one of ['elu', 'gelu',
            'hardshrink', 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid',
            'softmax', 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax',
            'hsigmoid']. Default: None, means identical transformation.

    Returns:
        A callable object of Activation.

    Raises:
        KeyError: When parameter `act` is not in the optional range.

    Examples:

        from paddleseg.models.common.activation import Activation

        relu = Activation("relu")
        print(relu)
        # <class 'paddle.nn.layer.activation.ReLU'>

        sigmoid = Activation("sigmoid")
        print(sigmoid)
        # <class 'paddle.nn.layer.activation.Sigmoid'>

        not_exit_one = Activation("not_exit_one")
        # KeyError: "not_exit_one does not exist in the current dict_keys(['elu', 'gelu', 'hardshrink',
        # 'tanh', 'hardtanh', 'prelu', 'relu', 'relu6', 'selu', 'leakyrelu', 'sigmoid', 'softmax',
        # 'softplus', 'softshrink', 'softsign', 'tanhshrink', 'logsigmoid', 'logsoftmax', 'hsigmoid'])"
    """

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.layer.activation.{}()".format(
                    act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x

def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)

class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

@register
class MultiSegHead(nn.Layer):
    def __init__(self,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 num_classes=4,
                 min_kept=[26000, 26000, 26000],
                 coef=[1, 1, 1],
                 pretrained=None):
        super().__init__()

        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))
        self.coef = coef
        self.loss_layers = []
        for min_k in min_kept:
            self.loss_layers.append(OhemCrossEntropyLoss(min_kept=min_k))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, neck_feats, inputs=None, x_hw=None):
        neck_feats = neck_feats[::-1]
        if self.training:
            logit_list = []
            for x, seg_head in zip(neck_feats, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]

            # loss computation
            labels = inputs['mask'].astype('int64')
            loss_list = []
            for i in range(len(logit_list)):
                logits = logit_list[i]
                loss_list.append(self.coef[i] * self.loss_layers[i](logits, labels))
            return sum(loss_list)
        else:
            x = self.seg_heads[0](neck_feats[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logits = paddle.argmax(x, axis=1)
            return logits

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def post_process(self, logits):
        logits = paddle.argmax(logits, axis=1).numpy().squeeze()

        preds = []
        for cid in range(8, 11):
            mask = (logits == cid-7).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                points = contour.reshape(-1,2)
                if(len(points) >= 3):
                    cate_area = np.array([cid, int(area)]).reshape(1, 2)
                    pred = np.concatenate((cate_area, points), axis=0)
                    preds.append(pred)
        return preds
