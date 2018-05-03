from glob import glob
import json
import cv2
import numpy as np



selected_classes = ['car']
dataset = 'trainval'
count = 0
max_count = 0
num = 48433
ind = 0
len_s = 80
img_array = np.zeros([num, 3, 224, 224])
label_array = np.zeros([num, len_s, 28 * 28 + 1])
label_index_array = np.zeros([num, len_s])
files = glob('img/'+dataset+'/*/*.png')
for file in files:
    json_file = 'label' + file[3:-15] + 'gtFine_polygons.json'
    json_object = json.load(open(json_file))
    h = json_object['imgHeight']
    w = json_object['imgWidth']
    objects = json_object['objects']
    for obj in objects:
        #         if obj['label'] == 'truck':
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
            I = cv2.imread(file)
            I_obj = I[min_row:max_row, min_col:max_col, :]
            I_obj_new = cv2.resize(I_obj, (224, 224))
            cv2.imwrite('car_img/'+dataset+'/'+ str(count) + '.png', I_obj_new)
            I_obj = np.rollaxis(I_obj_new, 2)
            img_array[count, :, :, :] = I_obj
            point_count = 0
            last_index_a = -1
            last_index_b = -1
            dic = {}
            dic['polygon'] = []
            #             len_p = len(obj['polygon'])
            for p_count, points in enumerate(obj['polygon']):
                index_a = (points[0] - min_col) * scale_w
                index_b = (points[1] - min_row) * scale_h
                index_a = np.maximum(0, np.minimum(223, index_a))
                index_b = np.maximum(0, np.minimum(223, index_b))
                dic['polygon'].append([index_a, index_b])
            with open('car_label/' +dataset+'/' + str(count) + '.json', 'w') as jsonfile:
                json.dump(dic, jsonfile)
            #                 index = index_b*28 + index_a
            #                 if abs(index_a-last_index_a)<=1 and abs(index_b-last_index_b)<=1:
            #                     continue
            #                 last_index_a = index_a
            #                 last_index_b = index_b
            #                 label_array[count,point_count,index]=1
            #                 label_index_array[count,point_count]=index
            #                 point_count += 1
            #             for kkk in range(point_count,len_s):
            #                 index = 28*28
            #                 label_array[count,kkk,index]=1
            #                 label_index_array[count,kkk]=index
            #             label_array[count,point_count,28*28]=1

            #             if max_count < point_count:
            #                 max_count = point_count
            count += 1
    #             plt.imshow(I[min_c[1]:max_c[1],min_c[0]:max_c[0],:])
    #             for points in obj['polygon']:
    #                 plt.scatter(points[0]-min_c[0],points[1]-min_c[1])
    #             plt.show()

    #             plt.imshow(I_obj_new)
    #             for points in obj['polygon']:
    #                 plt.imshow(I_obj_new)
    #                 plt.scatter((points[0]-min_col)*scale_w,(points[1]-min_row)*scale_h)
    #                 plt.show()
    if (count / 10000 >= ind):
        ind += 1
        print(count)

#     if(count>100):
#         break
print(count)
print(max_count)
# print(files[3][3:-15])