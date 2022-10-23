import os, json
import numpy as np
from tqdm import tqdm

EPSILON = 1e-6

def calc_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    overlap_x = max(min(x1+w1, x2+w2) - max(x1, x2), 0)
    overlap_y = max(min(y1+h1, y2+h2) - max(y1, y2), 0)
    inter = overlap_x * overlap_y
    union = w1*h1 + w2*h2 - inter
    return inter / (union + EPSILON)

def split_val_test(gd_path, pd_path, val_prop=0.5):
    data = json.load(open(gd_path, 'r'))
    # split images
    images = data['images']
    N = len(images)
    val_num = int(N * val_prop)
    val_images = images[:val_num]
    test_images = images[val_num:]

    # same categories
    categories = data['categories']

    # split annotations
    val_iids = [image['id'] for image in val_images]
    test_iids = [image['id'] for image in test_images]
    val_annos = []
    test_annos = []
    for anno in data['annotations']:
        if(anno['image_id'] in val_iids):
            val_annos.append(anno)
        elif(anno['image_id'] in test_iids):
            test_annos.append(anno)
        else:
            raise "anno not in val & test?"

    val_data = {'images': val_images,
                'categories': categories,
                'annotations': val_annos }
    test_data = {'images': test_images,
                 'categories': categories,
                 'annotations': test_annos }
    print("Val %d images %d annotations" % (len(val_images), len(val_annos)))
    print("Test %d images %d annotations" % (len(test_images), len(test_annos)))

    annos = json.load(open(pd_path, 'r'))
    # split 
    val_annos = []
    test_annos = []
    for anno in annos:
        if(anno['image_id'] in val_iids):
            val_annos.append(anno)
        elif(anno['image_id'] in test_iids):
            test_annos.append(anno)
        else:
            raise "anno not in val & test?"
    
    return val_data, test_data, val_annos, test_annos

def select_best_scores(gt_data, preds):
    cid2names = dict()
    for cate in gt_data['categories']:
        cid2names[cate['id']] = cate['name']

    iid2annos = dict()
    for anno in gt_data['annotations']:
        image_id = anno['image_id']
        if(image_id in iid2annos):
            iid2annos[image_id].append(anno)
        else:
            iid2annos[image_id] = [anno]

    iid2preds = dict()
    for pred in preds:
        image_id = pred['image_id']
        if(image_id in iid2preds):
            iid2preds[image_id].append(pred)
        else:
            iid2preds[image_id] = [pred]

    for cid, category in cid2names.items():
        best_thresh, best_area_th, best_F1 = select_best_score(filter_cate(iid2annos, cid), filter_cate(iid2preds, cid))
        print("Best score for %s : %.5f @ %.3f %d" % (category, best_F1, best_thresh, best_area_th))
        with open('det_threshes.txt', 'a') as f:
            f.write('%.2f\n' % best_thresh)
        

def filter_cate(iid2annos, cid):
    data = dict()
    for image_id, annos in iid2annos.items():
        for anno in annos:
            if(anno['category_id'] == cid):
                if(image_id in data):
                    data[image_id].append(anno)
                else:
                    data[image_id] = [anno]
    return data

def get_TP_FP_FN(annos, preds):
    gt_bboxes = [anno['bbox'] for anno in annos]
    pd_bboxes = [pred['bbox'] for pred in preds]
    M, N = len(gt_bboxes), len(pd_bboxes)
    ious = np.zeros((M,N), dtype=np.float)
    for i in range(M):
        for j in range(N):
            ious[i, j] = calc_iou(gt_bboxes[i], pd_bboxes[j])
    method = 1
    if(method == 1):
        if(M == 0 or N == 0):
            TP = 0
        else:
            TP = np.sum(np.max(ious, axis=1) > 0.5)
        FN = M - TP
        FP = N - TP
    else:
        if(M == 0 or N == 0):
            TP = 0
        else:
            TP = np.sum(np.max(ious, axis=0) > 0.5)
        FP = N - TP
        if(M == 0):
            FN = 0
        elif(N == 0):
            FN = M
        else:
            FN = np.sum(np.max(ious, axis=1) <=0.5)
    return TP, FP, FN

def select_best_score(iid2annos, iid2preds):
    all_iids = set(list(iid2annos.keys()) + list(iid2preds.keys()))
    best_F1 = 0
    best_thresh = 0
    best_area_th = 0
    for thresh in tqdm(range(1, 98)):
        thresh /= 100
        for area_thresh in range(0, 1): 
            TP = FP = FN = 0
            for image_id in all_iids:
                preds = []
                for pred in iid2preds.get(image_id, []):
                    if(pred['score'] > thresh and pred['bbox'][2]*pred['bbox'][3] > area_thresh):
                        preds.append(pred)
                TP_, FP_, FN_ = get_TP_FP_FN(iid2annos.get(image_id, []), preds)
                TP += TP_
                FP += FP_
                FN += FN_
            prec = TP / float(TP + FP + EPSILON)
            recall = TP / float(TP + FN + EPSILON)
            F1 = 2*prec*recall/(prec + recall + EPSILON)
            if(F1 > best_F1):
                best_F1 = F1
                best_thresh = thresh
                best_area_th = area_thresh
    return best_thresh, best_area_th, best_F1

def calc_F1(iid2annos, iid2preds):
    all_iids = set(list(iid2annos.keys()) + list(iid2preds.keys()))
        
    TP = FP = FN = 0
    for image_id in all_iids:
        preds = iid2preds.get(image_id, [])
        TP_, FP_, FN_ = get_TP_FP_FN(iid2annos.get(image_id, []), preds)
        TP += TP_
        FP += FP_
        FN += FN_
    prec = TP / float(TP + FP + EPSILON)
    recall = TP / float(TP + FN + EPSILON)
    F1 = 2*prec*recall/(prec + recall + EPSILON)
    return F1

def test_F1(gt_data, preds, threshes):
    cid2names = dict()
    for cate in gt_data['categories']:
        cid2names[cate['id']] = cate['name']

    iid2annos = dict()
    for anno in gt_data['annotations']:
        image_id = anno['image_id']
        if(image_id in iid2annos):
            iid2annos[image_id].append(anno)
        else:
            iid2annos[image_id] = [anno]

    iid2preds = dict()
    for pred in preds:
        cid = pred['category_id']
        image_id = pred['image_id']
        if(pred['score'] < threshes[cid2names[cid]]):
            continue
        if(image_id in iid2preds):
            iid2preds[image_id].append(pred)
        else:
            iid2preds[image_id] = [pred]

    for cid, category in cid2names.items():
        F1 = calc_F1(filter_cate(iid2annos, cid), filter_cate(iid2preds, cid))
        print("Test %s : %g" % (category, F1))

if __name__ == '__main__':
    gt = json.load(open('./dataset/val_det.json', 'r'))
    pd = json.load(open('bbox.json', 'r'))
    select_best_scores(gt, pd)
    exit(0)
