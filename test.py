from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.utils.data
from tensorboardX import SummaryWriter
import argparse
from data import load_data
from model import PolygonNet
from PIL import Image, ImageDraw
import numpy as np
import json
from glob import glob

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpu_id', nargs='+',type=int)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num', type=int, default=45000)
parser.add_argument('--test_mode', type=str, default='full')
parser.add_argument('--model', type=str, default='./save/model.pth')
args = parser.parse_args()

devices=args.gpu_id
print(devices)
batch_size = args.batch_size
num = args.num

torch.cuda.set_device(devices[0])
net = PolygonNet(load_vgg=False)
net = nn.DataParallel(net,device_ids=devices)
net.load_state_dict(torch.load(args.model))
net.cuda()
print('finished')

dtype = torch.cuda.FloatTensor
dtype_t = torch.cuda.LongTensor
    
if args.test_mode == 'small':

    Dataloader = load_data(num, 'trainval', 600, batch_size)
    len_dl = len(Dataloader)
    print(len_dl)

    nu = 0
    de = 0
    for step, data in enumerate(Dataloader):
        labels = data[4].numpy()
        xx = Variable(data[0].type(dtype))
        re = net.module.test(xx,60)
        for i in range(batch_size):
            labels_p = re.cpu().numpy()[i]
            vertices1 = []
            vertices2 = []
            for label in labels_p:
                if (label == 784):
                    break
                vertex = ((label % 28) * 8, (label / 28) * 8)
                vertices1.append(vertex)
            for label in labels[i]:
                if (label == 784):
                    break
                vertex = ((label % 28) * 8, (label / 28) * 8)
                vertices2.append(vertex)

            img = Image.fromarray(data[0][i].numpy().astype(np.uint8))
            drw = ImageDraw.Draw(img, 'RGBA')
            drw.polygon(vertices1, (255, 0, 0, 125))
            img.save('saveimg/small/'+str(step)+'_'+str(i)+'_pred.png', 'PNG')

            img = Image.fromarray(data[0][i].numpy().astype(np.uint8))
            drw = ImageDraw.Draw(img, 'RGBA')
            drw.polygon(vertices2, (255, 0, 0, 125))
            img.save('saveimg/small/' + str(step) +'_'+str(i)+ '_gt.png', 'PNG')


            img1 = Image.new('L', (224, 224), 0)
            ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
            mask1 = np.array(img1)
            img2 = Image.new('L', (224, 224), 0)
            ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
            mask2 = np.array(img2)
            intersection = np.logical_and(mask1, mask2)
            union = np.logical_or(mask1, mask2)
            nu += np.sum(intersection)
            de += np.sum(union)
    print(nu * 1.0 / de)

elif args.test_mode == 'full':

    selected_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'rider', 'bus', 'train']

    iou = {}
    for cls in selected_classes:
        iou[cls] = 0

    count = 0
    files = glob('img/val/*/*.png')
    for ind, file in enumerate(files):
        json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
        json_object = json.load(open(json_file))
        h = json_object['imgHeight']
        w = json_object['imgWidth']
        objects = json_object['objects']
        img = Image.open(file)
        I = np.array(img)
        img_gt = Image.open(file)
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
                max_col = np.minimum(w, max_c[0] + h_extend)
                object_h = max_row - min_row
                object_w = max_col - min_col
                scale_h = 224.0 / object_h
                scale_w = 224.0 / object_w
                I_obj = I[min_row:max_row, min_col:max_col, :]
                I_obj_img = Image.fromarray(I_obj)
                I_obj_img = I_obj_img.resize((224,224),Image.BILINEAR)
                I_obj_new = np.array(I_obj_img)
                I_obj = np.rollaxis(I_obj_new, 2)
                I_obj = I_obj[np.newaxis,:,:,:]

                xx = Variable(torch.from_numpy(I_obj).type(dtype))
                re = net.module.test(xx, 60)
                labels_p = re.cpu().numpy()[0]
                vertices1 = []
                vertices2 = []
                for label in labels_p:
                    if (label == 784):
                        break
                    vertex = (((label % 28) * 8.0 +4) / scale_w + min_col, ((int(label / 28)) * 8.0 +4) / scale_h + min_row)
                    vertices1.append(vertex)

                drw = ImageDraw.Draw(img, 'RGBA')
                drw.polygon(vertices1, (255, 0, 0, 100))


                for points in obj['polygon']:
                    vertex = (points[0], points[1])
                    vertices2.append(vertex)

                drw_gt = ImageDraw.Draw(img_gt, 'RGBA')
                drw_gt.polygon(vertices2, (255, 0, 0, 100))

        count += 1
        img.save('saveimg/full/' + str(ind) + '_pred.png', 'PNG')
        img_gt.save('saveimg/full/' + str(ind) + '_gt.png', 'PNG')
        if count>num:
            break

    print(count)


