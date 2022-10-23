import os, json, cv2
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

EPSILON = 1e-6

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

    print("Reading prediction files ...")
    data = json.load(open(pd_path, 'r'))
    val_preds = []
    test_preds = []
    for pred in data:
        imageid = pred['image_id']
        if(imageid in val_iids):
            val_preds.append(pred)
        else:
            test_preds.append(pred)
    
    return val_data, test_data, val_preds, test_preds

def calc_iou(seg1, seg2):
    seg1 = [tuple(point) for point in np.array(seg1).reshape(-1,2)]
    seg2 = [tuple(point) for point in np.array(seg2).reshape(-1,2)]
    polygon1 = Polygon(seg1)
    polygon2 = Polygon(seg2)
    try:
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        return intersect / (union+EPSILON)
    except:
        return 0

def select_best_scores(gt_data, preds, by_score=True):
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
        best_area, best_F1 = select_best_score_fast(filter_cate(iid2annos, cid), 
                                                    filter_cate(iid2preds, cid))
        with open('det_threshes.txt', 'a') as f:
            f.write(str(best_area))
            f.write('\n')
        print("Best F1 for %s : %.5f @ area: %d" % (category, best_F1, best_area))

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
    for anno in annos:
        assert(len(anno['segmentation']) == 1)
    gt_segs = [np.array(anno['segmentation'][0]).reshape(-1,2) for anno in annos]
    pd_segs = [np.array(pred['segmentation'][0]).reshape(-1,2) for pred in preds]
    M, N = len(gt_segs), len(pd_segs)
    ious = np.zeros((M,N), dtype=np.float)
    for i in range(M):
        for j in range(N):
            ious[i, j] = calc_iou(gt_segs[i], pd_segs[j])
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

def select_best_score_fast(iid2annos, iid2preds):
    pred_areas = []
    for iid, preds in iid2preds.items():
        pred_areas += [pred['count'] for pred in preds]
    pred_areas = np.array(pred_areas)

    max_N = max([len(elem) for elem in iid2preds.values()])
    M = sum([len(elem) for elem in iid2annos.values()])
    ious = np.zeros((M, max_N), dtype=np.float)
    areas = np.zeros((M, max_N), dtype=np.float)
    y = 0
    for image_id in iid2annos.keys():
        annos = iid2annos.get(image_id, [])
        preds = iid2preds.get(image_id, [])
        for anno in annos:
            for x, pred in enumerate(preds):
                iou = calc_iou(anno['segmentation'], pred['segmentation'])
                ious[y, x] = iou
                areas[y, x] = pred['count']
            y += 1
    assert(y == M)

    best_F1 = 0
    best_area = 0
    for area_thresh in tqdm(range(300, 2000, 20)):
        ious[areas <= area_thresh] = 0.
        TP = np.sum(np.max(ious, axis=1) > 0.5)
        FP = np.sum(pred_areas > area_thresh) - TP
        FN = M - TP

        prec = TP / float(TP + FP + EPSILON)
        recall = TP / float(TP + FN + EPSILON)
        F1 = 2*prec*recall/(prec + recall + EPSILON)
        if(F1 > best_F1):
            best_F1 = F1
            best_area = area_thresh
    return best_area, best_F1

def select_best_score_old(iid2annos, iid2preds):
    all_iids = set(list(iid2annos.keys()) + list(iid2preds.keys()))
    best_F1 = 0
    best_area = 0
    for area in tqdm(range(300, 2000, 20)):
        TP = FP = FN = 0
        for image_id in all_iids:
            preds = [pred for pred in iid2preds.get(image_id, []) if pred['count'] > area]
            TP_, FP_, FN_ = get_TP_FP_FN(iid2annos.get(image_id, []), 
                                         preds)
            TP += TP_
            FP += FP_
            FN += FN_
        prec = TP / float(TP + FP + EPSILON)
        recall = TP / float(TP + FN + EPSILON)
        F1 = 2*prec*recall/(prec + recall + EPSILON)
        if(F1 > best_F1):
            best_F1 = F1
            best_area = area
    return best_area, best_F1

if __name__ == '__main__':
    val_gt, test_gt, val_preds, test_preds = split_val_test('dataset/val_seg.json', 'seg.json', val_prop=1.0)
    select_best_scores(val_gt, val_preds, by_score=False)
