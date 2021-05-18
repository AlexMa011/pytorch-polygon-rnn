# pytorch-polygon-rnn
Pytorch implementation of [Polygon-RNN](http://www.cs.toronto.edu/polyrnn/poly_cvpr17/). 
Notice that I use another way to handle the first vertex instead of training another model as in the paper.
I will not maintain the project, please refer to [Polygon-RNN++](https://github.com/fidler-lab/polyrnn-pp) for better experience.

### Difference with the original paper

1. I use two virtual starting vertices to handle the first vertex as in the image captioning.

2. I add a LSTM layer after the ConvLSTM layers since I need the output to be  D\*D+1 dimension to handle the end symbol.

### How to train and test

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
python train.py --gpu_id 0 1 2 --batch_size 8  --lr 0.0001
```

4. Run the following command to validate.
```
python validate.py --gpu_id 0 1 2 --batch_size 8
```

5. Run the following command to test.
```
python test.py --gpu_id 0 --model ./save/model.pth
```

Now you can easily change configurations on default_config.yaml.
