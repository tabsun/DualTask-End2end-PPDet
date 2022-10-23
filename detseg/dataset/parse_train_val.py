import os, json, random, cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def parse_set(names, root, filter=True, image_id_start=0):
    categories = [{'id': 1,
                   'name': 'car',
        	   'supercategory': 'car'},
                  {'id': 2,
                   'name': 'vehicle',
        	   'supercategory': 'vehicle'},
                  {'id': 3,
        	   'name': 'person',
        	   'supercategory': 'person'},
                  {'id': 4,
        	   'name': 'red_light',
        	   'supercategory': 'red_light'},
                  {'id': 5,
                   'name': 'yellow_light',
        	   'supercategory': 'yellow_light'},
                  {'id': 6,
        	   'name': 'green_light',
        	   'supercategory': 'green_light'},
                  {'id': 7,
        	   'name': 'light',
        	   'supercategory': 'light'}
        	   ]
    if(not os.path.exists('label_mask')):
        os.makedirs('label_mask')
    images = []
    annos = []
    for id, name in tqdm(enumerate(names)):
        id += image_id_start
        image_path = os.path.join(root, 'images', name+'.jpg')
        width, height = Image.open(image_path).size
        images.append({'id': id,
                       'file_name': name+'.jpg',
        	       'height': height,
        	       'width': width})
        data = json.load(open(os.path.join(root, 'label', name+'.json'), 'r'))
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        for item in data:
            cid = item['type']
            if(cid <= 7):
                #cid = min(4, cid)
                x, y, w, h = item['x'], item['y'], item['width'], item['height']
                if(w * h < 20 and filter):
                    continue
                bbox = [x, y, w, h]
                segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                annos.append({'id': len(annos),
                              'image_id': id,
                              'category_id': cid,
                              'segmentation': segmentation,
                              'area': w*h,
                              'bbox': bbox,
                              'iscrowd': 0})
            else:
                cid -= 7
                points = np.array([round(elem) for elem in item['segmentation'][0]]).reshape(-1,2)
                mask = cv2.fillPoly(mask, [points], (cid, cid, cid))
        with open(os.path.join('label_mask', name+'.npy'), 'wb') as f:
            np.save(f, mask[:, :, 0])
    data = {'images': images,
            'categories': categories,
            'annotations': annos }
    return data, len(names)+image_id_start
        
if __name__ == '__main__':
    root = './all_data'

    train_image_ids = [line.strip() for line in open('train_image_ids.txt', 'r').readlines()]
    val_image_ids = [line.strip() for line in open('val_image_ids.txt', 'r').readlines()]

    print("Parsing train dataset...")
    train_data, id_count = parse_set(train_image_ids, root)
    with open('train_det.json', 'w') as f:
        json.dump(train_data, f)

    print("Parsing val dataset...")
    val_data, _ = parse_set(val_image_ids, root, filter=False, image_id_start=id_count)
    with open('val_det.json', 'w') as f:
        json.dump(val_data, f)

    val_data = json.load(open('val_det.json', 'r'))

    print("Parsing val seg annotation...")
    images = val_data['images']
    categories = [{'id': 8,
                   'name': 'solid_line',
        	   'supercategory': 'solid_line'},
                  {'id': 9,
                   'name': 'dot_line',
        	   'supercategory': 'dot_line'},
                  {'id': 10,
        	   'name': 'zebra_line',
        	   'supercategory': 'zebra_line'}]
    annos = []
    for image in images:
        image_id = image['id']
        data = json.load(open(os.path.join(root, 'label', image['file_name'].replace('.jpg', '.json')), 'r'))
        for item in data:
            if(item['type'] > 7):
                annos.append( {'id': len(annos),
                               'image_id': image_id, 
                               'category_id': item['type'],
                               'segmentation': item['segmentation']} )
    data = {'images': images, 'categories': categories, 'annotations': annos}
    with open('val_seg.json', 'w') as f:
        json.dump(data, f)
