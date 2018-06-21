# pytorch-polygon-rnn
Pytorch implementation of [Polygon-RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/). 
Notice that I use another way to handle the first vertex instead of training another model as in the paper.


### Download and Preprocess Data

1. Download data from [CityScapes](https://www.cityscapes-dataset.com/downloads/), organize
the image files and annotation json files as follows:



```
img
├── train
│   ├── cityname1
│   │   ├── pic.png
│   │   ├── ...
│   ├── cityname2
│   │   ├── pic.png
│   │   ├── ...
├── val
│   ├── cityname
│   │   ├── pic.png
│   │   ├── ...
├── test
│   ├── cityname
│   │   ├── pic.png
│   │   ├── ...
```

```
label
├── train
│   ├── cityname1
│   │   ├── annotation.json
│   │   ├── ...
│   ├── cityname2
│   │   ├── annotation.json
│   │   ├── ...
├── val
│   ├── cityname
│   │   ├── annotation.json
│   │   ├── ...
├── test
│   ├── cityname
│   │   ├── annotation.json
│   │   ├── ...
```

The png files and the json files should have corresponding same name.

Execute the following command to make directories for new data and save models:
```
mkdir -p new_img/(train/val/test)
mkdir -p new_label/(train/val/test)
mkdir save
```

2. Run the following command to generate data for train/validation/test.
```
python generate_data.py --data train/val/test
```

3. Run the following command to train.
```
python train.py --gpu_id 0 1 2 --batch_size 4 --pretrained False --lr 0.001
```

4. Run the following command to test.
```
python test.py --gpu_id 0 1 2 --mode full/small --model ./save/model.pth
