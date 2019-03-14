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

#### Single model performace
| single model           | privare LB|
| ---------------- |  ---- |
|resnet101_fold0_256x512|0.9696|
|seresnet101_fold0_256x512|0.9691|
|seresnext101_fold0_256x512|0.9692|
|resnet101_fold0_512x512|0.9682|
|seresnet101_fold0_512x512|0.9664|
|seresnext101_fold0_512x512|-|

#### Single model performace with pseudo labeling
I generate a pseudo label list containing 1.5k samples when I reached 0.940 in public LB, and I kept using this list till the competition ended. I used the bottleneck feature of the arcface model (my baseline model) to calculate cosine distance of train test images. For those few shot classes (less than 2 samples), I choose 0.65 as the threshold to filter high confidence samples.  I think it will be better result using 0.970 LB model to find pseudo label.

| single model           | privare LB|
| ---------------- |  ---- |
|resnet101_fold0_256x512|0.9705|
|seresnet101_fold0_256x512|0.9704|
|seresnext101_fold0_256x512|-|

#### Model ensemble performace
| single model           | privare LB|
| ---------------- |  ---- |
|resnet101_seresnet101_seresnext101_fold0_256x512|0.97113|
|resnet101_seresnet101_seresnext101_fold0_512x512_pseudo|0.97072|
|10 models(final submisson)|0.97209|

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

train resnet101 512x512 fold 0：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=train --model=resnet101 --image_h=512 --image_w=512 --fold_index=0 --batch_size=128
```

predict resnet101 256x512 fold 0 model：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=test --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128 --pretrained_mode=max_valid_model.pth
```

train resnet101 256x512 fold 0 with pseudo labeling：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=train --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128 --is_pseudo=True
```

predict resnet101 256x512 fold 0 model with pseudo labeling：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode=test --model=resnet101 --image_h=256 --image_w=512 --fold_index=0 --batch_size=128 --is_pseudo=True --pretrained_mode=max_valid_model.pth
```

#### Final Ensemble
the final submission is the weight average result of 10 ckpts
```
python ensemble.py
```



















