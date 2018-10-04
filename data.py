import json

import numpy as np
import torch
from PIL import Image
from torch import utils
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class newdataset(Dataset):
    def __init__(self, data_num, data_set, len_s, transform=None):
        self.num = data_num
        self.dataset = data_set
        self.length = len_s
        self.transform = transform
        # stuff

    def __getitem__(self, index):
        # stuff
        img_name = 'new_img/{}/{}.png'.format(self.dataset, index)
        label_name = 'new_label/{}/{}.json'.format(self.dataset, index)
        try:
            img = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            return None
        assert not (img is None)
        json_file = json.load(open(label_name))
        point_num = len(json_file['polygon'])
        polygon = np.array(json_file['polygon'])
        point_count = 2
        #         img_array = np.zeros([data_num, 3, 224, 224])
        label_array = np.zeros([self.length, 28 * 28 + 3])
        label_index_array = np.zeros([self.length])
        if point_num < self.length - 3:
            for points in polygon:
                index_a = int(points[0] / 8)
                index_b = int(points[1] / 8)
                index = index_b * 28 + index_a
                label_array[point_count, index] = 1
                label_index_array[point_count] = index
                point_count += 1
            label_array[point_count, 28 * 28] = 1
            label_index_array[point_count] = 28 * 28
            for kkk in range(point_count + 1, self.length):
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
                label_array[kkk, index] = 1
                label_index_array[kkk] = index
        else:
            scale = point_num * 1.0 / (self.length - 3)
            index_list = (np.arange(0, self.length - 3) * scale).astype(int)
            for points in polygon[index_list]:
                index_a = int(points[0] / 8)
                index_b = int(points[1] / 8)
                index = index_b * 28 + index_a
                label_array[point_count, index] = 1
                label_index_array[point_count] = index
                point_count += 1
            for kkk in range(point_count, self.length):
                index = 28 * 28
                label_array[kkk, index] = 1
                label_index_array[kkk] = index

        if self.transform is not None:
            img = self.transform(img)
        #         stride = self.length - 2
        return (img, label_array[2], label_array[:-2], label_array[1:-1],
                label_index_array[2:])

    def __len__(self):
        return self.num  # of how many examples(images?) you have


def load_data(data_num, data_set, len_s, batch_size):
    trans = transforms.ToTensor()
    datas = newdataset(data_num, data_set, len_s, trans)
    Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size,
                                             shuffle=True, drop_last=False)
    return Dataloader
