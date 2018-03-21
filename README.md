# LightNet
This repository contains the code (in PyTorch) for: "LightNet: Light-weight Networks for Semantic Image Segmentation
" (underway)  by Huijun Liu @ TU Braunschweig.

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Discussions](#discussions)
5. [Contacts](#contacts)

## Introduction
As exact understanding the surrounding scene is very important for the navigation and driving decision of the self-
driving car, semantic segmentation is a significant part of the modern autonomous driving system. Nowadays, deep
fully convolutional networks (FCNs) have a very significant effect on semantic segmentation, but most of the relevant
research has focused on improving segmentation accuracy rather than model computation efficiency. The autonomous
driving system is often based on embedded devices, where computing and storage resources are relatively limited.
In this paper we describe several light-weight networks based on MobileNetV2, Additionally, we introduce concurrent
Spatial-Channel Sequeeze & Excitation (SCSE) and Receptive Field Block (RFB) to the proposed network. We measure our performance on Cityscapes pixel-level segmentation, and achieve 70.72% class mIoU. We evaluate the trade-offs between mIoU, and number of operations measured by
multiply-add (MAdd), as well as the number of parameters.

### Network Architecture
underway...

## Usage

### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(0.3.0+)](http://pytorch.org)
- [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [tqdm](https://github.com/tqdm/tqdm)
- [inplace_abn](https://github.com/mapillary/inplace_abn)
- [CityscapesScripts](https://github.com/mcordts/cityscapesScripts)
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)

### Train
As an example, use the following command to train a LightNet on Cityscapes 
or Mapillary Vistas Dataset:

```
> python3 train_mobile.py
> python3 train_mobile_mvd.py 
> python3 train_share.py 
> python3 train_mixscale.py 
> python3 train_shuffle.py 
> python3 train_inplace.py 
```

### Evaluation
We take the Cityscapes model trained above as an example.

To evaluate the trained model:

```
> python3 deploy_model.py
> python3 evaluation_cityscapes.py
> python3 demo_cityscapes.py
```

### Other Options
We also include Mixed-scale DenseNet, ShuffleNetPlus, SE-DPShuffleNet, Wide-ResNet implementation in this repository.  


## Results

### Results on Cityscapes (Pixel-level/Semantic Segmentation)

| Model | GFLOPs | Params |gtFine/gtCoarse| mIoU Classes(val./test) | mIoU Cat.(val./test) | Result(*.cvs) | Pytorch Model&Checkpoint |
|---|---|---|---|---|---|---|---|
|MobileNetV2Plus|117.1?|8.3M|Yes/No|73.89/70.72|88.72/87.64|[GoogleDrive](https://drive.google.com/open?id=1b1NJhe4sQ126d7xqg-d9mf8WNTstAoER)|[GoogleDrive](https://drive.google.com/open?id=19s7mdCJqTgZ17hgN7_t17sP-RM_FibmW)|
|RF-MobileNetV2Plus|87.6?|8.6M|Yes/No|72.37/70.68|88.31/88.27|[GoogleDrive](https://drive.google.com/open?id=1JmB5KNmMV92yk5qtjwZnX-ZOhU35Pk6Y)|[GoogleDrive](https://drive.google.com/open?id=1QKLJ7u3DKKOTrMGQCFOprqQZWVrmWQm7)|
|SE-WResNet|?|?M|Yes/No|*/77.00|*/89.70|/|/|


## Contact
ansleliu@gmail.com  
h.liu@tu-braunschweig.de

We are working on the implementation on other frameworks.
Any discussions or concerns are welcomed!

## References
underway...
