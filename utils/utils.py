from PIL import Image, ImageDraw
import numpy as np
import torch

def iou(vertices1, vertices2, h ,w):
    '''
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    '''
    img1 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
    mask1 = np.array(img1)
    img2 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
    mask2 = np.array(img2)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    nu = np.sum(intersection)
    de = np.sum(union)
    if de!=0:
        return nu*1.0/de, nu, de
    else:
        return 0, nu, de

def label2vertex(labels):
    '''
    convert 1D labels to 2D vertices coordinates
    :param labels: 1D labels
    :return: 2D vertices coordinates: [(x1, y1),(x2,y2),...]
    '''
    vertices = []
    for label in labels:
        if (label == 784):
            break
        vertex = ((label % 28) * 8, (label / 28) * 8)
        vertices.append(vertex)
    return vertices

def getbboxfromkps(kps,h,w):
    '''

    :param kps:
    :return:
    '''
    min_c = np.min(np.array(kps), axis=0)
    max_c = np.max(np.array(kps), axis=0)
    object_h = max_c[1] - min_c[1]
    object_w = max_c[0] - min_c[0]
    h_extend = int(round(0.1 * object_h))
    w_extend = int(round(0.1 * object_w))
    min_row = np.maximum(0, min_c[1] - h_extend)
    min_col = np.maximum(0, min_c[0] - w_extend)
    max_row = np.minimum(h, max_c[1] + h_extend)
    max_col = np.minimum(w, max_c[0] + w_extend)
    return (min_row,min_col,max_row,max_col)

def img2tensor(img):
    '''

    :param img:
    :return:
    '''
    img = np.rollaxis(img,2,0)
    return torch.from_numpy(img)

def tensor2img(tensor):
    '''

    :param tensor:
    :return:
    '''
    img = (tensor.numpy()*255).astype('uint8')
    img = np.rollaxis(img,0,3)
    return img