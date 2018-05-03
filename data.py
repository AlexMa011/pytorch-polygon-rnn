from glob import glob
import numpy as np
import json
from PIL import Image
import torch
from torch import utils

def load_data(data_num, data_set, len_s, batch_size):
    files = glob('new_img/'+data_set+'/*.png')
    ind = 0
    count = 0
    img_array = np.zeros([data_num, 3, 224, 224])
    label_array = np.zeros([data_num, len_s, 28 * 28 + 3])
    label_index_array = np.zeros([data_num, len_s])
    label_array[:, 0, 28 * 28 + 1] = 1
    label_array[:, 1, 28 * 28 + 2] = 1
    label_index_array[:, 0] = 28 * 28 + 1
    label_index_array[:, 1] = 28 * 28 + 2
    
    for file in files:
        I = np.rollaxis(np.array(Image.open(file)), 2)
        img_array[count] = I
        json_file = json.load(open('new_label' + file[7:-4] + '.json'))
        point_num = len(json_file['polygon'])
        polygon = np.array(json_file['polygon'])
        point_count = 2
        if point_num < len_s - 3:
            for points in polygon:
                index_a = int(points[0] / 8)
                index_b = int(points[1] / 8)
                index = index_b * 28 + index_a
                label_array[count, point_count, index] = 1
                label_index_array[count, point_count] = index
                point_count += 1
            label_array[count, point_count, 28 * 28] = 1
            label_index_array[count, point_count] = 28 * 28
            for kkk in range(point_count + 1, len_s):
                if kkk % (point_num + 3) == point_num + 2:
                    index = 28 * 28
                elif kkk % (point_num + 3) == 0:
                    index = 28 * 28 + 1
                elif kkk % (point_num + 3) == 1:
                    index = 28 * 28 + 2
                else:
                    index_a = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
                    index_b = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
                    index = index_b * 28 + index_a
                label_array[count, kkk, index] = 1
                label_index_array[count, kkk] = index
        else:
            scale = point_num * 1.0 / (len_s - 3)
            index_list = (np.arange(0, len_s - 3) * scale).astype(int)
            for points in polygon[index_list]:
                index_a = int(points[0] / 8)
                index_b = int(points[1] / 8)
                index = index_b * 28 + index_a
                label_array[count, point_count, index] = 1
                label_index_array[count, point_count] = index
                point_count += 1
            for kkk in range(point_count, len_s):
                index = 28 * 28
                label_array[count, kkk, index] = 1
                label_index_array[count, kkk] = index
        count += 1
        if count >= data_num:
            break

    
        if (count / 10000 >= ind):
            ind += 1
            print('Load {} data '.format(count))
    

    print('Load all data!')

    stride = len_s - 2
    input1_array = label_array[:, 2, :]
    input2_array = label_array[:, 0:0 + stride, :]
    input3_array = label_array[:, 1:1 + stride, :]
    target_array = label_index_array[:, 2:2 + stride]
    x = torch.from_numpy(img_array[:])
    x1 = torch.from_numpy(input1_array)
    x2 = torch.from_numpy(input2_array)
    x3 = torch.from_numpy(input3_array)
    t = torch.from_numpy(target_array).contiguous().long()
    print(x.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(t.shape)

    Dataset = utils.data.TensorDataset(x, x1, x2, x3, t)
    Dataloader = utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return Dataloader