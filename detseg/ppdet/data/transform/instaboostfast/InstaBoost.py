import numpy as np
import os, json, cv2
import random
from tqdm import tqdm
from copy import deepcopy
from .get_instance_group import extract, extract_single_instance
from .affine_transform import transform_image, transform_image_onlyxy, transform_annotation
from .config import InstaBoostConfig
from .exceptions import *
from .pointByHeatmap import paste_position
from PIL import Image, ImageEnhance, ImageOps, ImageFile

class_names = [
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag']

def calculate_scale(pt, size):
    x, y = pt
    w, h = size
    return abs(x - w/2) / w * 2

def crop_and_resize_pillar(image, w, h):
    ori_h, ori_w = image.shape[:2]
    if(h >= ori_h):
        return cv2.resize(image, (w, h))
    else:
        image = cv2.resize(image, (w, ori_h))
        return image[ori_h-h:, :, :]

def get_border(mask, x):
    offset = 0
    h, w = mask.shape[:2]
    while(True):
        if(x + offset < w):
            top, bottom = get_border_single(mask, x+offset)
            if(top < bottom):
                return top, bottom
        if(x - offset > 0):
            top, bottom = get_border_single(mask, x-offset)
            if(top < bottom):
                return top, bottom
        offset += 1
        if(x + offset >= w and x - offset < 0):
            return h//2, h//2+1

def get_border_single(mask, x):
    h, w = mask.shape[:2]
    assert(x >= 0 and x < w)
    col = mask[:, x].flatten()
    top = 0
    bottom = h-1
    while(col[top] < 128 and top < bottom):
        top += 1
    while(col[bottom] < 128 and bottom > top):
        bottom -= 1
    return top, bottom


def fix_anno(anno_, center, box_size, size):
    anno = deepcopy(anno_)
    cx, cy = center
    w, h = box_size
    ori_w, ori_h = size
    seg = np.array(anno['segmentation'][0]).reshape(-1,2).astype(np.float)
    bx, by, bw, bh = anno['bbox']

    seg[:, 0] -= (bx + bw/2)
    seg[:, 1] -= (by + bh/2)
    assert(abs(w/bw - h/bh) < 0.3)
    scale = w / bw
    seg *= scale

    seg[:,0] += cx
    seg[:,1] += cy
    seg[:, 0] = np.clip(seg[:,0], 0, ori_w)
    seg[:, 1] = np.clip(seg[:,1], 0, ori_h)

    anno['segmentation'] = [[int(x) for x in seg.flatten()]]
    #xmin, xmax = cx-w/2, cx+w/2
    #ymin, ymax = cy-h/2, cy+h/2
    #xmin = max(0, xmin)
    #xmax = min(w, xmax)
    #ymin = max(0, ymin)
    #ymax = min(h, ymax)
    xmin, ymin = np.min(seg, axis=0)
    xmax, ymax = np.max(seg, axis=0)
    anno['bbox'] = [xmin, ymin, xmax-xmin-1, ymax-ymin-1]
    return anno

def identity_transform():
    t = dict()
    t['s'] = 1
    t['tx'] = 0
    t['ty'] = 0
    t['theta'] = 0
    return t
        
class InstaBoostInstances:
    def __init__(self, anno_file, image_dir, seg_dir, config_dict):
        """
        :param action_candidate: tuple of action candidates. 'normal', 'horizontal', 'vertical', 'skip' are supported
        :param action_prob: tuple of corresponding action probabilities. Should be the same length as action_candidate
        :param scale: tuple of (min scale, max scale)
        :param dx: the maximum x-axis shift will be  (instance width) / dx
        :param dy: the maximum y-axis shift will be  (instance height) / dy
        :param theta: tuple of (min rotation degree, max rotation degree)
        :param color_prob: the probability of images for color augmentation
        :param heatmap_flag: whether to use heatmap guided
        """
        with open(anno_file, 'r') as f:
            info = json.load(f)

        annotations = info['annotations']
        image_infos = info['images']
        categories  = info['categories']

        imageid2info = dict()
        self.fname2imageid = dict()
        for image_info in image_infos:
            imageid2info[image_info['id']] = image_info
            self.fname2imageid[image_info['file_name']] = image_info['id']

        self.instances_of_each_cate = dict()
        self.heatmap_of_each_cate = dict()
        _, categories = zip(*sorted(zip([cate['id'] for cate in categories], [cate['name'] for cate in categories])))
        for category in categories:
            config = config_dict[category]
            sync_expect = sum([num*prob for num, prob in zip(config['sync_nums_only_obj'],
                                                             config['sync_prob_only_obj'])])
            #sync_expect += sum([num*prob for num, prob in zip(config['sync_nums_with_pillar'],
            #                                                 config['sync_prob_with_pillar'])])
            if(sync_expect <= 0):
                continue

            category_id = categories.index(category) + 1
            if(config['sync_heatmap_flag_only_obj']):
                print("Calculating heatmap for %s..." % category)
                heatmap_h = 720
                heatmap_w = 1280
                heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float)
                for anno in tqdm(annotations):
                    if(anno['category_id'] == category_id):
                        x, y, w, h = anno['bbox']
                        image_info = imageid2info[anno['image_id']]
                        scale_x = float(heatmap_w) / image_info['width']
                        scale_y = float(heatmap_h) / image_info['height']
                        x = max(0, int(x*scale_x))
                        y = max(0, int(y*scale_y))
                        w = min(heatmap_w-x, int(w*scale_x))
                        h = min(heatmap_h-y, int(h*scale_y))
                        heatmap[y:y+h, x:x+w] += 1
                        
                self.heatmap_of_each_cate[category] = heatmap / np.sum(heatmap)

            instances = []
            
            print("Extracting instances for %s..." % category)
            N = 2000
            areas = [anno['bbox'][2] * anno['bbox'][3] for anno in annotations if anno['category_id'] == category_id]
            area_thresh = max(50, sorted(areas, reverse=True)[min(len(areas)-1, 3*N)])
            for anno in tqdm(annotations):
                if(anno['category_id'] != category_id):
                    continue
                if(anno['bbox'][2] * anno['bbox'][3] < area_thresh):
                    continue
                # extract 2000 instances at most for each category
                if(len(instances) > N):
                    break
                ori_ann = {'category_id': category_id, 'segmentation': anno['segmentation'], 'bbox': anno['bbox']}
                file_name = imageid2info[anno['image_id']]['file_name']
                ori_img = cv2.imread(os.path.join(image_dir, file_name))
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                instance, _ = extract_single_instance(ori_ann, ori_img, flip_prob=0.5)
                instances.append([instance, ori_img.shape[:2], ori_ann])

            self.instances_of_each_cate[category] = instances
            print("Get %d instances of category %s" % (len(instances), category))

        # # DEBUG
        # for category, heatmap in self.heatmap_of_each_cate.items():
        #     heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        #     heatmap = cv2.merge([heatmap] * 3)
        #     cv2.imwrite('heatmap_%s.jpg'%category, heatmap)
        
        # # semantic segmentation results are used to generate heatmap
        # self.seg_non_mask = np.load(os.path.join(seg_dir, 'mask.npy'))
        # self.imageid2segmask = dict()
        # for fname in os.listdir(seg_dir):
        #     if(fname.endswith('npy') and fname.replace('npy','jpg') in self.fname2imageid):
        #         imageid = self.fname2imageid[fname.replace('npy','jpg')]
        #         self.imageid2segmask[imageid] = os.path.join(seg_dir, fname)
           
        self.config_dict = config_dict
        self.categories = categories

    def get_sync_data_only_obj(self, ori_anns, ori_img, image_info=None):
        #start_t = cv2.getTickCount()
        if len(ori_img.shape) == 2 or ori_img.shape[2] == 1:  # gray scale
            ori_img = cv2.merge([ori_img] * 3)
    
        ori_anns_bak = deepcopy(ori_anns)

        # # get heat map and seg_non_mask(to avoid some error in segmentation)
        # seg_file = self.imageid2segmask[int(image_info['im_id'].flatten()[0])]
        # with open(seg_file, 'rb') as f:
        #     seg_mask = np.load(f)
        # seg_non_mask = (cv2.resize(self.seg_non_mask.astype(np.uint8)*255, tuple(seg_mask.shape[:2][::-1])) < 128)

        ori_h, ori_w = ori_img.shape[:2]
        sync_instances_list = []
        transforms_list = []
        sync_anns = deepcopy(ori_anns)
        for index, category in enumerate(self.categories):
            #if(category in ['pillar', 'stand_eye', 'left_beam', 'right_beam']):
            #    continue
            cfg = self.config_dict[category]
            sync_nums = cfg['sync_nums_only_obj']
            sync_prob = cfg['sync_prob_only_obj']
            use_heatmap = cfg['sync_heatmap_flag_only_obj']
            sync_num = np.random.choice(sync_nums, p=sync_prob)
            if(sync_num == 0):
                continue

            anchors = []
            if(use_heatmap):
                heatmap = self.heatmap_of_each_cate[category]
                heatmap_h, heatmap_w = heatmap.shape[:2]
                index = np.random.choice(np.arange(heatmap_w * heatmap_h), size=sync_num, replace=False, p=heatmap.flatten())
                for i in index:
                    anchor_x = (i % heatmap_w) * ori_w // heatmap_w
                    anchor_y = (i // heatmap_w) * ori_h // heatmap_h
                    anchors.append([anchor_x, anchor_y])
            else:
                for _ in range(sync_num):
                    anchor_x = int(random.random() * ori_w)
                    anchor_y = int(random.random() * ori_h)
                    anchors.append([anchor_x, anchor_y])

            for anchor_x, anchor_y in anchors:
                instance, fake_size, fake_anno = deepcopy(random.choice(self.instances_of_each_cate[category]))
                scale_x = float(fake_anno['bbox'][2]) / fake_size[1]
                scale_y = float(fake_anno['bbox'][3]) / fake_size[0]
                eye_w = int(ori_w * scale_x)
                eye_h = int(ori_h * scale_y)
                instance = cv2.resize(instance.astype(np.uint8), (eye_w, eye_h)).astype(np.int32)
                transform = identity_transform()
                transform['tx'] = anchor_x 
                transform['ty'] = anchor_y - eye_h // 2
                sync_instances_list.append(instance)
                transforms_list.append(transform)
                # recalculate the fake anno to sync_anno
                sync_anno = fix_anno(fake_anno, (transform['tx'], transform['ty']), (eye_w, eye_h), (ori_w, ori_h))
                sync_anno['category_id'] = self.categories.index(category)
                sync_anns.append(sync_anno)
                        
        try:
            background = ori_img.copy()
            t = cv2.getTickCount()
            sync_img = transform_image_onlyxy(background, sync_instances_list, transforms_list)
            t = cv2.getTickCount() - t
            #print("Transform image: %gms with %d" % (t*1000/cv2.getTickFrequency(), len(sync_instances_list)))
        except (AnnError, TrimapError):
            sync_anns = ori_anns_bak
            sync_img = ori_img
    
        #for ann in sync_anns:
        #    cid = ann['category_id']
        #t = cv2.getTickCount() - start_t
        #print("Sync time : %gms" % (t*1000/cv2.getTickFrequency()))
        return sync_anns, sync_img

    def get_trans_data(self, ori_anns, ori_img):
        for ann in ori_anns:
            cid = ann['category_id']
            assert(cid >= 0 and cid <= 2)

        if len(ori_img.shape) == 2 or ori_img.shape[2] == 1:  # gray scale
            ori_img = cv2.merge([ori_img] * 3)
    
        trans_anns = []
        trans_cfgs = []
        trans_ids = []
        for index, ann in enumerate(ori_anns):
            category = self.categories[ann['category_id']]
            cfg = self.config_dict[category]
            aug_flag = np.random.choice([0,1],p=[1-cfg.aug_prob, cfg.aug_prob])
            if(aug_flag):
                trans_anns.append(ann)
                trans_cfgs.append(cfg)
                trans_ids.append(index)

        trans_anns_bak = deepcopy(trans_anns)
        try:
            background, trans_instances_list, transforms_list, bbox_list = extract(trans_anns, ori_img, trans_cfgs)
            assert background.shape == ori_img.shape, 'Background and original image shape mismatch'
            
            #if config.heatmap_flag:
            #    heatmap_guided_pos_list = paste_position(ori_anns, ori_img, groupidx_list, groupbnds_list)
            #    for i in range(len(heatmap_guided_pos_list)):
            #        heatmap_guided_pos = heatmap_guided_pos_list[i]
           
            #        if heatmap_guided_pos[0] != -1:
            #            transforms_list[i]['tx'] = heatmap_guided_pos[1]
            #            transforms_list[i]['ty'] = heatmap_guided_pos[0]
    
            groupidx_list = [[x] for x in range(len(bbox_list))]
            new_img = transform_image(background, trans_instances_list, transforms_list)
            new_ann = transform_annotation(trans_anns, transforms_list, bbox_list, groupidx_list,
                                           background.shape[1], background.shape[0])
        except (AnnError, TrimapError):
            new_ann = trans_anns_bak
            new_img = ori_img
    
        for new_id, ori_id in enumerate(trans_ids):
            if(new_ann[new_id] is not None):
                ori_anns[ori_id] = new_ann[new_id]

        return ori_anns, new_img
