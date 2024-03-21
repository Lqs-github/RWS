## RWS: Refined Weak Slice for Semantic Segmentation Enhancement

Yunbo Rao,  Qingsong Lv, Andrei Sharf, Zhanglin Cheng*

Paper: [10.1109/TCSVT.2024.3361463](https://doi.org/10.1109/tcsvt.2024.3361463)

<div align="center">
<br>
	<img width="100%" alt="Fig2_Flowchart" src="./Figs/Fig2_Flowchart.png">
</div>

## Abstract
> Interpretation of predictions made by Convolutional Neural Networks (CNNs) is a rapidly growing field of research. A common approach involves enhancing semantic segmentation predictions through the generation of heatmaps that illustrate the significance of individual pixels in the segmentation. Nevertheless, the selection of beneficial features from these heatmaps remains a challenge. This is because the introduced information often contains interfering factors such as mutual features between different objects, background, and insufficient heat map resolution which often diminish its effectiveness. To overcome these limitations, we introduce Refined Weak Slices (RWS). Our main idea is to identify low attention regions in heat maps i.e. **weak slices**, in conjunction with segmentation accuracy,  and utilize them to select effective features across different DNN layers, to enhance segmentation. We then seamlessly integrate these features back into the CNN, thus **refining** and enhancing the semantic segmentation result with selected features. Through extensive experiments, we demonstrate that incorporating the RWS module into state-of-the-art methods yields a notable improvement in the average mIoU by 2.84% on benchmark datasets (VOC 2012, COCOStuff, ADE20K, Cityscapes) for both ResNet-101 and ResNet-50 architectures. Furthermore, we achieve a maximum improvement of 5.8% with a single CNN. Overall, the combination of RWS and CNNs exhibits excellent performance in image segmentation tasks. 

## Preparations

#### 1. Download VOC 2012 dataset

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```

After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `data/VOCdevkit/VOC2012`. The directory structure should thus be

```
data
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        ├── JPEGImages
            ├──Action
            ├──Layout
            ├──Main
            └──Segmentation
                ├──train.txt
                ├──val.txt
        ├── SegmentationClass
        └── SegmentationObject
```

We organized the VOC2012 dataset by individual categories. Each class's list of labels and image names are stored separately for training and testing of individual types in `FWM_datalist_20Class`.  

Put the contents of  `train_type_list.txt ` and  `val_type_list.txt ` for each category into  `train.txt ` and  `val.txt ` under ` ./data/VOCdevkit/VOC2012/JPEGImages/Segmentation ` when training the types. Image labels into  `./data/VOCdevkit/VOC2012/SegmentationClass `.

If you want to replace your training dataset, update the image name list in train.txt and val.txt and add the corresponding image labels within the SegmentationClass folders.

#### 2. Clone RWS

```bash
A link to the code will be posted after the review. The code is uploaded as an attachment at this time.
```

#### 3. Download pre-training weights 

* deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth

* deeplabv3_resnet101: https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth

  Rename the download weight to ```deeplabv3_resnet50_coco.pth```

  Rename the download weight to ```deeplabv3_resnet101_coco.pth```

  We use deeplabv3_resnet50 as an example in the code.

#### 4. Requirements

```
numpy
Pillow
torch>=1.7.1
torchvision>=0.8.2
ttach
tqdm
opencv-python
matplotlib
scikit-learn
grad-cam
```

### Train

Just Run train.py

### Results

DeepLab v3, PCAA, SC-CAM and RWS have trained on VOC 2012 dataset with the unenhanced dataset and tested on DAVIS 2017.

<div align="center">
<b>Visualization</b>. <i>Left:</i> DeepLab v3. <i>Right:</i> RWS.
<br>
  <img width="45%" alt="DeepLab" src="./Figs/case_dogs.gif">
  <img width="45%" alt="Seg prediction" src="./Figs/case_dog.gif">
<div align="center">
<b>Visualization</b>. <i>Left:</i> PCAA. <i>Right:</i> RWS.
<br>
  <img width="45%" alt="PCAA" src="./Figs/cars.gif">
  <img width="45%" alt="Seg prediction" src="./Figs/car.gif">
<div align="center">
<b>Visualization</b>. <i>Left:</i> SC-CAM. <i>Right:</i> RWS.
<br>
  <img width="45%" alt="SC-CAM" src="./Figs/case_boats.gif">
  <img width="45%" alt="Seg prediction" src="./Figs/case_boat.gif">

### Guiding DNNs to improve themselves

<div align="center">
<br>
	<img width="100%" alt="Fig8_Effective" src="./Figs/Fig8_Effective.png">
</div>

#### 1. Replacement of DNNs

```bash
git clone the code you want to enhance (with CNN clusters)
```

#### 2. Calling all CNN convolutional layers

Modify the convolutional layer called here to be the last convolutional layer of the DLSNs.

```
target_layers = [model.model.backbone.layer4]
```

#### 3. Show the heat map of missing weights for each convolution

Modify line 321 of train.py to be True.

```python
CAM_VIS = True
```

#### 4. Your ablation experiment or new structure replacement

Start your experiments and see how much the convolutional heat map of omega changes for each layer, which can be used to guide your structural improvements. In addition, the converging RoI makes it easier for you to find the bottleneck of the segmentation.

<center>
    <img src="./Figs/Small_1.jpg" alt="Small_1.jpg" style="zoom:76.5%;"/>
    <img src="./Figs/Small_2.jpg" alt="Small_1.jpg" style="zoom:82.5%;"/>
    <center style="text-decoration">Small RoI (Person)</center>
</center>
