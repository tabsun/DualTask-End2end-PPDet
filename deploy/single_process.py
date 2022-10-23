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

import codecs
import os
import time
import sys
import math
import threading
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.inference import Config
from paddle.inference import create_predictor

class PredictConfig():
    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))

def get_test_images(infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
        # print(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
        images.append(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images

class End2end(object):
    def __init__(self,
                 det_model_dir,
                 block_dir,
                 device_id):
        self.predictor = self.load_predictor(det_model_dir, block_dir, device_id)
        self.det_threshes = [float(line.strip()) for line in open(os.path.join(det_model_dir, 'det_threshes.txt'), 'r').readlines()]
        
    def load_predictor(self, model_dir, block_dir, device_id):
        config = Config(
            os.path.join(model_dir, 'model.pdmodel'),
            os.path.join(model_dir, 'model.pdiparams'))
        # initial GPU memory(M), device ID
        config.enable_use_gpu(8000, device_id)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
        # disable print log when predict
        config.disable_glog_info()
        # enable shared memory
        config.enable_memory_optim()
        # disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        # init run
        #for block in os.listdir(block_dir):
        #    if(block.endswith('.npy')):
        #        input_name = block.split('.')[0]
        #        input_tensor = predictor.get_input_handle(input_name)
        #        input_tensor.copy_from_cpu(np.load(os.path.join(block_dir, block)))
        #predictor.run()
 
        return predictor

    def predict(self, image_ids, inputs):
        input_names = self.predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        logits_tensor = self.predictor.get_output_handle(output_names[2])
        ori_shapes_tensor = self.predictor.get_output_handle(output_names[3])
        bboxes = boxes_tensor.copy_to_cpu()
        logits = logits_tensor.copy_to_cpu()
        ori_shapes = ori_shapes_tensor.copy_to_cpu()
        
        # DEBUG
        #if(14571 in image_ids):
        #    print(image_ids)
        #    np.save('static.npy', logits)
        #    exit(0)

        bboxes_num = len(bboxes)
        bbox_results = bboxes[:, 2:]
        id_results =   bboxes[:, 0]
        score_results = bboxes[:, 1]

        preds = []
        # Parse detection results
        for idx in range(bboxes_num):
            input_id = idx // 100
            image_id = image_ids[input_id]
            cid = int(id_results[idx])
            if float(score_results[idx]) > self.det_threshes[cid]:
                preds.append({'image_id': image_id,
                              'x': float(bbox_results[idx][0]),
                              'y': float(bbox_results[idx][1]),
                              'width': float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                              'height': float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                              'type': int(id_results[idx]) + 1,
                              'segmentation': []})

        # Parse segmentation results
        for idx in range(logits.shape[0]):
            logit = logits[idx]
            ori_h, ori_w = ori_shapes[idx]
            h, w = logit.shape
            assert(h == 900 and w == 1600)
            scale_h, scale_w = float(ori_h)/h, float(ori_w)/w
            
            image_id = image_ids[idx]
            for cid in range(8, 11):
                mask = (logit == cid-7).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    points = contour.reshape(-1,2).astype(np.float)
                    points[:, 0] *= scale_w
                    points[:, 1] *= scale_h
                    xmin, ymin = np.min(points, axis=0)
                    xmax, ymax = np.max(points, axis=0)
                    #points = np.round(points).astype(np.int)
                    if(len(points) >= 3 and area > self.det_threshes[cid-1]):
                        points = [points.flatten().tolist()]
                        preds.append({'type': int(cid),
                                      'x': xmin,
                                      'y': ymin,
                                      'width': xmax-xmin,
                                      'height': ymax-ymin, 
                                      'image_id': int(image_id),
                                      'segmentation': points } )

        return preds

class OriDataLoader:
    def __init__(self, image_infos, size=(640,640), std_mean=True, worker_num=8, batch_size=8, max_queue_size=8, cuda_vars=None, check_size=(0, 10000)):
        self.output_queue = [] 
        self.input_h = size[0]
        self.input_w = size[1]
        self.std_mean = std_mean
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.left_worker_num = worker_num
        self.cuda_vars = cuda_vars
        self.check_size = check_size

        batch = math.ceil(len(image_infos) / worker_num)
        for i in range(worker_num):
            cur_image_infos = image_infos[i*batch:(i+1)*batch]
            threading.Thread(target=self.create_one_worker, args=(cur_image_infos,)).start()
    
    def create_one_worker(self, image_infos):
        N = len(image_infos)
        image_ids = []
        images = []
        im_shapes = []
        scale_factors = []

        for i in range(N):
            image = image_infos[i]
            image_id = image['image_id']
            file_path = image['file_path']

            input_h = self.input_h
            input_w = self.input_w
            image = cv2.imread(file_path)
            h, w = image.shape[:2]
            if(h > self.check_size[0] and h < self.check_size[1]):
                image = cv2.resize(image, None, None, fx=float(input_w)/w, fy=float(input_h)/h, interpolation=2)
                image_ids.append(image_id)
                images.append(image)
                im_shapes.append(np.array([input_h, input_w]))
                scale_factors.append(np.array([float(input_h)/h, float(input_w)/w]))
            
                if(len(image_ids) == self.batch_size):
                    images = np.array(images).astype(np.float32) / 255.
                    if(self.std_mean):
                        images[:,:,:,0] = (images[:,:,:,0] - 0.406) / 0.225
                        images[:,:,:,1] = (images[:,:,:,1] - 0.456) / 0.224
                        images[:,:,:,2] = (images[:,:,:,2] - 0.485) / 0.229
                    else:
                        images = (images - 0.5) / 0.5
                    images = np.transpose(images[:,:,:,::-1], axes=(0,3,1,2))
                    inputs = {'image': images,
                              'im_shape': np.array(im_shapes).astype(np.float32).reshape(-1,2),
                              'scale_factor': np.array(scale_factors).astype(np.float32).reshape(-1,2) }
                    if(self.cuda_vars is None):
                        for k, v in inputs.items():
                            inputs[k] = paddle.to_tensor(v).cuda()
                    else:
                        for k in self.cuda_vars:
                            inputs[k] = paddle.to_tensor(inputs[k]).cuda()

                    item = [image_ids, inputs]
                    while(len(self.output_queue) > self.max_queue_size):
                        time.sleep(1)
                    self.output_queue.append(item)

                    image_ids = []
                    images = []
                    im_shapes = []
                    scale_factors = []
            
        if(image_ids and i == N-1):
            images = np.array(images).astype(np.float32) / 255.
            if(self.std_mean):
                images[:,:,:,0] = (images[:,:,:,0] - 0.406) / 0.225
                images[:,:,:,1] = (images[:,:,:,1] - 0.456) / 0.224
                images[:,:,:,2] = (images[:,:,:,2] - 0.485) / 0.229
            else:
                images = (images - 0.5) / 0.5
            images = np.transpose(images[:,:,:,::-1], axes=(0,3,1,2))
            inputs = {'image': images,
                      'im_shape': np.array(im_shapes).astype(np.float32).reshape(-1,2),
                      'scale_factor': np.array(scale_factors).astype(np.float32).reshape(-1,2) }
            if(self.cuda_vars is None):
                for k, v in inputs.items():
                    inputs[k] = paddle.to_tensor(v).cuda()
            else:
                for k in self.cuda_vars:
                    inputs[k] = paddle.to_tensor(inputs[k]).cuda()

            item = [image_ids, inputs]
            while(len(self.output_queue) > self.max_queue_size):
                time.sleep(1)
            self.output_queue.append(item)

        self.left_worker_num -= 1

    def empty(self):
        return len(self.output_queue) == 0

    def finish(self):
        return self.left_worker_num == 0 and self.empty()

    def get_item(self):
        if(not self.empty()):
            return self.output_queue.pop(0)
        else:
            return None

if __name__ == '__main__':
    start_t = cv2.getTickCount()
    dist.init_parallel_env()
    det_model_path = "./model/v8"

    paddle.enable_static()
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]
    pid = int(sys.argv[3])
    device_id = pid % 4
    process_num = int(sys.argv[4])
    paddle.set_device('gpu:%d'%device_id)

    # Get all image infos
    image_infos = []
    if(infer_txt.endswith('.txt')):
        image_paths = get_test_images(infer_txt)
        for image_path in image_paths:
            image_id = int(os.path.basename(image_path).split('.')[0])
            image_infos.append({'image_id': image_id, 'file_path': image_path})
    elif(infer_txt.endswith('.json')):
        image_dir = '../../copy_data/images/'
        data = json.load(open(infer_txt, 'r'))
        for image in data['images']:
            image_id = image['id']
            image_path = os.path.join(image_dir, image['file_name'])
            image_infos.append({'image_id': image_id, 'file_path': image_path})
    else:
        raise "argument error: only txt or json file allowed"
    print("Get %d samples to test" % len(image_infos))

    image_infos = sorted(image_infos, key=lambda x:x['image_id'])
    step = math.ceil(len(image_infos) / process_num)
    image_infos = image_infos[pid*step:pid*step + step]
    print("Get %d sample in single process" % len(image_infos))
    

    preds = []
    dataloader = OriDataLoader(image_infos, size=(900,1600), worker_num=4, std_mean=True, batch_size=4, cuda_vars=[], max_queue_size=16, check_size=(0, 1000000))
    detector = End2end(det_model_path, './blocks', device_id)

    cnt = 0
    while(not dataloader.finish()):
        if(dataloader.empty()):
            time.sleep(0.1)
            continue
        image_ids, inputs = dataloader.get_item()

        one_t = cv2.getTickCount()
        preds += detector.predict(image_ids, inputs)
        one_t = cv2.getTickCount() - one_t
        cnt += 1
        print("%d: %d %gms" % (cnt, len(image_ids), one_t*1000/cv2.getTickFrequency()))
    print("Finish detection use time %gs" % ((cv2.getTickCount()-start_t)/cv2.getTickFrequency())) 

    with open(result_path+'_%d.json'%pid, 'w') as f:
        json.dump(preds, f)
    print("Get %d predictions" % len(preds))

    t = cv2.getTickCount() - start_t
    print("Total time %gs" % (t/cv2.getTickFrequency()))
