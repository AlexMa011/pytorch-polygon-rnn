import argparse
import json
import os
from glob import glob

import numpy as np
import torch.utils.data
from PIL import Image, ImageDraw
from torch import nn
from torch.autograd import Variable

from config_tools import get_config
from model import PolygonNet
from utils.utils import img2tensor
from utils.utils import iou, getbboxfromkps


def test(net, dataset, num=float('inf')):
    '''
    Test on validation dataset
    :param net: net to evaluate
    :param mode: full image or cropped image
    :param Dataloader: data to evaluate
    :return:
    '''

    dtype = torch.cuda.FloatTensor
    dtype_t = torch.cuda.LongTensor

    dir_name = 'save_img/test/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle',
                        'rider', 'bus', 'train']

    iou_score = {}
    nu = {}
    de = {}
    for cls in selected_classes:
        iou_score[cls] = 0.0
        nu[cls] = 0.0
        de[cls] = 0.0

    count = 0
    files = glob('img/{}/*/*.png'.format(dataset))
    for ind, file in enumerate(files):
        json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
        json_object = json.load(open(json_file))
        h = json_object['imgHeight']
        w = json_object['imgWidth']
        objects = json_object['objects']
        img = Image.open(file).convert('RGB')
        I = np.array(img)
        img_gt = Image.open(file).convert('RGB')
        for obj in objects:
            if obj['label'] in selected_classes:
                min_row, min_col, max_row, max_col = getbboxfromkps(
                    obj['polygon'], h, w)
                object_h = max_row - min_row
                object_w = max_col - min_col
                scale_h = 224.0 / object_h
                scale_w = 224.0 / object_w
                I_obj = I[min_row:max_row, min_col:max_col, :]
                I_obj_img = Image.fromarray(I_obj)
                I_obj_img = I_obj_img.resize((224, 224), Image.BILINEAR)
                I_obj_new = np.array(I_obj_img)
                xx = img2tensor(I_obj_new)
                xx = xx.unsqueeze(0).type(dtype)

                xx = Variable(xx)
                re = net.module.test(xx, 60)
                labels_p = re.cpu().numpy()[0]
                vertices1 = []
                vertices2 = []
                color = [np.random.randint(0, 255) for _ in range(3)]
                color += [100]
                color = tuple(color)
                for label in labels_p:
                    if (label == 784):
                        break
                    vertex = (
                        ((label % 28) * 8.0 + 4) / scale_w + min_col, (
                            (int(label / 28)) * 8.0 + 4) / scale_h + min_row)
                    vertices1.append(vertex)

                try:
                    drw = ImageDraw.Draw(img, 'RGBA')
                    drw.polygon(vertices1, color)
                except TypeError:
                    continue

                for points in obj['polygon']:
                    vertex = (points[0], points[1])
                    vertices2.append(vertex)

                drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
                drw_gt.polygon(vertices2, color)
                _, nu_this, de_this = iou(vertices1, vertices2, h, w)
                nu[obj['label']] += nu_this
                de[obj['label']] += de_this

        count += 1
        img.save(dir_name + str(ind) + '_pred.png', 'PNG')
        img_gt.save(dir_name + str(ind) + '_gt.png', 'PNG')
        if count >= num:
            break

    for cls in iou_score:
        iou_score[cls] = nu[cls] * 1.0 / de[cls] if de[cls] != 0 else 0
    return iou_score
    # print('number of pictures: {}'.format(count))
    # print('iou score:{}'.format(iou_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpu_id', nargs='+', type=int)
    parser.add_argument('--num', type=int, )
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--config', dest='config_file', help='Config File')
    args = parser.parse_args()

    config_from_args = args.__dict__
    config_file = config_from_args.pop('config_file')
    config = get_config('test', config_from_args, config_file)

    devices = config['gpu_id']
    num = config['num']
    dataset = config['dataset']
    model = config['model']

    print('gpus: {}'.format(devices))
    torch.cuda.set_device(devices[0])

    net = PolygonNet(load_vgg=False)
    net = nn.DataParallel(net, device_ids=devices)
    net.load_state_dict(torch.load(config['model']))
    net.cuda()
    print('Loading completed!')

    iou_score = test(net, dataset, num)
    print('iou score:{}'.format(iou_score))
