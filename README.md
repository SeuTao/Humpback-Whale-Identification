# Kaggle Humpback Whale Identification Challenge 2019 2nd place code
This is the source code for my part of the 2nd place solution to the [Humpback Whale Identification Challenge](https://www.kaggle.com/c/humpback-whale-identification) hosted by Kaggle.com. 

# End to End Whale Identification Model 

## Recent Update

**`2019.03.13`**: code upload.

#### Dependencies
- imgaug == 0.2.8
- opencv-python==3.4.2
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- torch==1.0.1.
- torchvision==0.2.2

## Solution Development
#### single model design

- Input: 256x512 or 512*512 cropped images;
- Backbone: resnet101, seresnet101, seresnext101;
- Loss function: arcface loss + triplet loss + focal loss;
- optimizer: adam with warm up lr strategy；
- Augmentation: blur,grayscale,noise,shear,rotate,perspective transform；
- Horizontal flip to create more ids -> 5004*2
- Pseudo Labeling


#### Path Setup
Set the following path to your own in ./process/data_helper.py
```
PJ_DIR = r'/Kaggle_Whale2019_2nd_place_solution'#project path
train_df = pd.read_csv('train.csv') #train.csv path
TRN_IMGS_DIR = '/train/'#train data path
TST_IMGS_DIR = '/test/' #test data path
```

#### Single Model Training
train resnet101 256x512 fold 0：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=train --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128
```

train resnet101 256x512 fold 0 with pseudo labeling：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=train --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128 --is_pseudo=True
```

predict resnet101 256x512 fold 0 model：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=test --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128 --pretrained_mode=max_valid_model.pth
```

#### Final Ensemble

```
python ensemble.py
```



















