# LightNet
## !!!New Repo!!! **[LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation](https://github.com/ansleliu/LightNetPlusPlus)**

This repository contains the code (in PyTorch) for: "LightNet: Light-weight Networks for Semantic Image Segmentation
" (underway) by Huijun Liu @ TU Braunschweig.

- **[MobileNetV2Plus](https://www.cityscapes-dataset.com/method-details/?submissionID=1157)**: Modified MobileNetV2<sup>[[1,8]](#references)</sup> + Spatial-Channel Sequeeze & Excitation (SCSE)<sup>[[6]](#references)</sup> + ASPP<sup>[[2,3]](#references)</sup> + Encoder-Decoder Arch.<sup>[[3]](#references)</sup> + InplaceABN<sup>[[4]](#references)</sup>.
- **[RF-MobileNetV2Plus](https://www.cityscapes-dataset.com/method-details/?submissionID=1172)**: Modified MobileNetV2 + SCSE + Receptive Field Block (RFB)<sup>[[5]](#references)</sup> + Encoder-Decoder Arch. + InplaceABN.
- **MobileNetV2Vortex**: Modified MobileNetV2 + SCSE + Vortex Pooling<sup>[[24]](#references)</sup> + Encoder-Decoder Arch. + InplaceABN.
- **MobileNetV2Share**: Split Image & Concat Features + Modified MobileNetV2 + SCSE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.
- **Mixed-scale DenseNet**: Modified Mixed-scale DenseNet<sup>[[11]](#references)</sup> + SCSE + ASPP/RFB + InplaceABN.
- **SE-WResNetV2**: Modified WResNetV2 (InplaceABN & SCSE/SE)<sup>[[4,6,7,13]](#references)</sup> + SCSE/SE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.
- **ShuffleNetPlus**: Modified ShuffleNet<sup>[[9]](#references)</sup> + SCSE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.

**!!!New!!!**: add **Vortex Pooling**<sup>[[24]](#references)</sup>

I no longer have GPUs to continue more experiments and model training (I **borrowed** 2 GPUs from the [Institute for Computer Graphics @ TU Braunschweig](https://www.cg.cs.tu-bs.de/) to complete preliminary experiments, so thank them and [Lukas Zhang](https://github.com/ZHHJemotion) here.), 
so if you like, welcome to experiment with other under-training models and my ideas!

<p align="center">
<a href="https://youtu.be/0g9zDGSRBi0" target="_blank">
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/cityscapes_demo_lightnet.gif" width="896" height="384" />
</a>
</p>

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [References](#references)
5. [Acknowledgement](#acknowledgement)

## Introduction
Semantic Segmentation is a significant part of the modern autonomous driving system, as exact understanding the surrounding scene is very important for the navigation and driving decision of the self-driving car. 
Nowadays, deep fully convolutional networks (FCNs) have a very significant effect on semantic segmentation, but most of the relevant researchs have focused on improving segmentation accuracy rather than model computation efficiency. 
However, the autonomous driving system is often based on embedded devices, where computing and storage resources are relatively limited. 
In this paper we describe several light-weight networks based on MobileNetV2, ShuffleNet and Mixed-scale DenseNet for semantic image segmentation task, Additionally, we introduce **GAN for data augmentation**<sup>[[17]](#references)</sup> ([pix2pixHD](https://github.com/NVIDIA/pix2pixHD)) concurrent **Spatial-Channel Sequeeze & Excitation** (SCSE) and **Receptive Field Block** (RFB) to the proposed network. 
We measure our performance on Cityscapes pixel-level segmentation, and achieve up to **70.72%** class mIoU and **88.27%** cat. mIoU. We evaluate the trade-offs between mIoU, and number of operations measured by multiply-add (MAdd), as well as the number of parameters.

### Network Architecture
<p align="center">
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/MobileNetV2Plus%26RF-MobileNetV2Plus.png" />
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/ASPP.png" />
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/Original-RFBlock.png" />
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/VortexPooling.png" />
<img src="https://github.com/ansleliu/LightNet/blob/master/resource/SCSE.png" />
</p>


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
We also include Mixed-scale DenseNet/RF-Mixed-scale DenseNet, ShuffleNetPlus/RFShuffleNetPlus, SE-DPShuffleNet, SE-Wide-ResNetV2 implementation in this repository.  


## Results
Mixed-scale DenseNet/RF-Mixed-scale DenseNet, ShuffleNetPlus/RFShuffleNetPlus, SE-DPShuffleNet, SE-Wide-ResNetV2 under-training (Ask a friend for help)

### Results on Cityscapes (Pixel-level/Semantic Segmentation)

| Model | GFLOPs | Params |gtFine/gtCoarse/GAN| mIoU Classes(val./test S.S*) | mIoU Cat.(val./test S.S*) | Result(*.cvs) | Pytorch Model&Checkpoint |
|---|---|---|---|---|---|---|---|
|MobileNetV2Plus|117.1?|8.3M|Yes/No/No|70.13/68.90|87.95/86.85|[GoogleDrive](https://drive.google.com/open?id=1D7maZzuunop_CJHeFIkuqv2gFEoxUanq)|/|
|MobileNetV2SDASPP|?|?M|Yes/No/Yes|73.17/70.09|87.98/87.84|[GoogleDrive](https://drive.google.com/open?id=1tiECuwuQ8S8rx4H94pkemqozX39Seusn)|[GoogleDrive](https://drive.google.com/open?id=1umpCqNk_2XovkvTTNKLct0p7P6NaN-0w)|
|[MobileNetV2Plus](https://www.cityscapes-dataset.com/method-details/?submissionID=1157)|117.1?|8.3M|Yes/No/Yes|73.89/**70.72**|88.72/87.64|[GoogleDrive](https://drive.google.com/open?id=1b1NJhe4sQ126d7xqg-d9mf8WNTstAoER)|[GoogleDrive](https://drive.google.com/open?id=19s7mdCJqTgZ17hgN7_t17sP-RM_FibmW)|
|[RF-MobileNetV2Plus](https://www.cityscapes-dataset.com/method-details/?submissionID=1172)|87.6?|8.6M|Yes/No/Yes|72.37/70.68|88.31/**88.27**|[GoogleDrive](https://drive.google.com/open?id=1JmB5KNmMV92yk5qtjwZnX-ZOhU35Pk6Y)|[GoogleDrive](https://drive.google.com/open?id=1QKLJ7u3DKKOTrMGQCFOprqQZWVrmWQm7)|
|ShuffleNetPlus|229.3?|15.3M|Yes/No/Yes|*|*|*|*|
|Mixed-scale DenseNet|49.9?|0.80M|Yes/No/Yes|*|*|*|*|
|SE-WResNetV2|?|?M|Yes/No/No|80.13/77.15|90.87/90.59|[GoogleDrive](https://drive.google.com/open?id=1MIJL6cfoBt3opcPWeNfudXLimE42Ow6_)|/|

* S.S.: Single Scale
  
## Contact
ansleliu@gmail.com  
h.liu@tu-braunschweig.de

Any discussions or concerns are welcomed!

## References
[1]. [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381v2)  
[2]. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)  
[3]. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v2)  
[4]. [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://arxiv.org/abs/1712.02616v2)  
[5]. [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767v2)  
[6]. [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579v1)  
[7]. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507v1)  
[8]. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1)  
[9]. [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)  
[10]. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407v1)  
[11]. [A mixed-scale dense convolutional neural network for image analysis](https://slidecam-camera.lbl.gov/static/asset/PNAS.pdf)  
[12]. [Dual Path Networks](https://arxiv.org/abs/1707.01629v2)  
[13]. [Wide Residual Networks](https://arxiv.org/abs/1605.07146v4)  
[14]. [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)  
[15]. [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/abs/1711.09224v1)  
[16]. [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323v2)  
[17]. [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585v1)  
[18]. [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983v5)  
[19]. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186v6)  
[20]. [Group Normalization](https://128.84.21.199/abs/1803.08494v1)  
[21]. [Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904v1)  
[22]. [ExFuse: Enhancing Feature Fusion for Semantic Segmentation](https://arxiv.org/abs/1804.03821v1)  
[23]. [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790v2)  
[24]. [Vortex Pooling: Improving Context Representation in Semantic Segmentation](https://arxiv.org/abs/1804.06242v1)  
  

# Acknowledgement

[0]. [Lukas Zhang](https://github.com/ZHHJemotion): Lend me GPUs.  
[1]. [meetshah1995](https://github.com/meetshah1995): [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).  
[2]. [ruinmessi](https://github.com/ruinmessi): [RFBNet](https://github.com/ruinmessi/RFBNet).  
[3]. [Jackson Huang](https://github.com/jaxony): [ShuffleNet](https://github.com/jaxony/ShuffleNet).  
[4]. [Ji Lin](https://github.com/tonylins): [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2).  
[5]. [ericsun99](https://github.com/ericsun99): [MobileNet-V2-Pytorch](https://github.com/ericsun99/MobileNet-V2-Pytorch).  
[6]. [Ross Wightman](https://github.com/rwightman): [pytorch-dpn-pretrained](https://github.com/rwightman/pytorch-dpn-pretrained).  
[7]. [mapillary](https://github.com/mapillary): [inplace_abn](https://github.com/mapillary/inplace_abn).  
[8]. [Cadene](https://github.com/Cadene): [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch).  
[9]. [Maxim Berman](https://github.com/bermanmaxim): [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax).  
