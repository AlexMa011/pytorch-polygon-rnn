import argparse
import json
import os
from glob import glob

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--data', type=str, default="train")
args = parser.parse_args()

selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'rider',
                    'bus', 'train']
dataset = args.data
files = glob('img/{}/*/*.png'.format(dataset))
count = 0
max_count = 0
ind = 0

img_dir = './new_img/{}'.format(dataset)
label_dir = './new_label/{}'.format(dataset)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)


for file in files:
    json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
    json_object = json.load(open(json_file))
    h = json_object['imgHeight']
    w = json_object['imgWidth']
    objects = json_object['objects']
    for obj in objects:
        if obj['label'] in selected_classes:
            min_c = np.min(np.array(obj['polygon']), axis=0)
            max_c = np.max(np.array(obj['polygon']), axis=0)
            object_h = max_c[1] - min_c[1]
            object_w = max_c[0] - min_c[0]
            h_extend = int(round(0.1 * object_h))
            w_extend = int(round(0.1 * object_w))
            min_row = np.maximum(0, min_c[1] - h_extend)
            min_col = np.maximum(0, min_c[0] - w_extend)
            max_row = np.minimum(h, max_c[1] + h_extend)
            max_col = np.minimum(w, max_c[0] + w_extend)
            object_h = max_row - min_row
            object_w = max_col - min_col
            scale_h = 224.0 / object_h
            scale_w = 224.0 / object_w
            I = Image.open(file)
            I_obj = I.crop(box=(min_col, min_row, max_col, max_row))
            I_obj_new = I_obj.resize((224, 224), Image.BILINEAR)
            I_obj_new.save('new_img/' + dataset + '/' + str(count) + '.png',
                           'PNG')
            point_count = 0
            last_index_a = -1
            last_index_b = -1
            dic = {}
            dic['polygon'] = []
            for p_count, points in enumerate(obj['polygon']):
                index_a = (points[0] - min_col) * scale_w
                index_b = (points[1] - min_row) * scale_h
                index_a = np.maximum(0, np.minimum(223, index_a))
                index_b = np.maximum(0, np.minimum(223, index_b))
                dic['polygon'].append([index_a, index_b])
            with open('new_label/' + dataset + '/' + str(count) + '.json',
                      'w') as jsonfile:
                json.dump(dic, jsonfile)
            count += 1

    if (count / 10000 >= ind):
        ind += 1
        print(count)

print(count)
print(max_count)
