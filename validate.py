import argparse
import os

import numpy as np
import torch.utils.data
from PIL import Image, ImageDraw
from torch import nn
from torch.autograd import Variable

from config_tools import get_config
from data import load_data
from model import PolygonNet
from utils.utils import iou, label2vertex
from utils.utils import tensor2img


def validate(net, Dataloader):
    '''
    Test on validation dataset
    :param net: net to evaluate
    :param Dataloader: data to evaluate
    :return:
    '''

    dtype = torch.cuda.FloatTensor
    dtype_t = torch.cuda.LongTensor

    dir_name = 'save_img/validate/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    len_dl = len(Dataloader)
    print(len_dl)

    nu = 0
    de = 0
    for step, data in enumerate(Dataloader):
        labels = data[4].numpy()
        xx = Variable(data[0].type(dtype))
        re = net.module.test(xx, 60)
        for i in range(len(re)):
            labels_p = re.cpu().numpy()[i]
            vertices1 = label2vertex(labels_p)
            vertices2 = label2vertex(labels[i])

            color = [np.random.randint(0, 255) for _ in range(3)]
            color += [100]
            color = tuple(color)

            img_array = tensor2img(data[0][i])

            img = Image.fromarray(img_array)
            drw = ImageDraw.Draw(img, 'RGBA')
            drw.polygon(vertices1, color)
            img.save(
                dir_name + str(step) + '_' + str(i) + '_pred.png',
                'PNG')

            img = Image.fromarray(img_array)
            drw = ImageDraw.Draw(img, 'RGBA')
            drw.polygon(vertices2, color)
            img.save(
                dir_name + str(step) + '_' + str(i) + '_gt.png',
                'PNG')

            _, nu_this, de_this = iou(vertices1, vertices2, 224, 224)
            nu += nu_this
            de += de_this

    print('iou: {}'.format(nu * 1.0 / de))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpu_id', nargs='+', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num', type=int, )
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--config', dest='config_file', help='Config File')
    args = parser.parse_args()

    config_from_args = args.__dict__
    config_file = config_from_args.pop('config_file')
    config = get_config('val', config_from_args, config_file)

    devices = config['gpu_id']
    batch_size = config['batch_size']
    num = config['num']
    dataset = config['dataset']
    model = config['model']

    print('gpus: {}'.format(devices))
    torch.cuda.set_device(devices[0])

    net = PolygonNet(load_vgg=False)
    net = nn.DataParallel(net, device_ids=devices)
    net.load_state_dict(torch.load(model))
    net.cuda()
    print('Loading completed!')

    Dataloader = load_data(num, dataset, 600, batch_size)
    validate(net, Dataloader)
