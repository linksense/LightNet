# LightNet
This repository contains the code (in PyTorch) for: "LightNet: Light-weight Networks for Semantic Image Segmentation
" (underway) by Huijun Liu @ TU Braunschweig.

- **MobileNetV2Plus**: Modified MobileNetV2[1] + Spatial-Channel Sequeeze & Excitation (SCSE) + ASPP + Encoder-Decoder Arch. + InplaceABN.
- **RF-MobileNetV2Plus**: Modified MobileNetV2 + SCSE + Receptive Field Block (RFB) + Encoder-Decoder Arch. + InplaceABN.
- **MobileNetV2Share**: Split Image & Concat Features + Modified MobileNetV2 + SCSE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.
- **Mixed-scale DenseNet**: Modified Mixed-scale DenseNet + SCSE + ASPP/RFB + InplaceABN.
- **SE-WResNetV2**: Modified WResNetV2 (InplaceABN & SCSE/SE) + SCSE/SE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.
- **ShuffleNetPlus**: Modified ShuffleNet + SCSE + ASPP/RFB + Encoder-Decoder Arch. + InplaceABN.

I no longer have GPUs to continue more experiments and model training (I **borrowed** 2 GPUs from the [Institute for Computer Graphics @ TU Braunschweig](https://www.cg.cs.tu-bs.de/) to complete preliminary experiments, so thank them and [Lukas Zhang](https://github.com/ZHHJemotion) here.), 
so if you like, welcome to experiment with other under-training models and my ideas!
## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Discussions](#discussions)
5. [Contacts](#contacts)

## Introduction
Semantic Segmentation is a significant part of the modern autonomous driving system, as exact understanding the surrounding scene is very important for the navigation and driving decision of the self-driving car. 
Nowadays, deep fully convolutional networks (FCNs) have a very significant effect on semantic segmentation, but most of the relevant researchs have focused on improving segmentation accuracy rather than model computation efficiency. 
However, the autonomous driving system is often based on embedded devices, where computing and storage resources are relatively limited. 
In this paper we describe several light-weight networks based on MobileNetV2, Additionally, we introduce GAN for data augmentation([pix2pixHD](https://github.com/NVIDIA/pix2pixHD)) concurrent Spatial-Channel Sequeeze & Excitation (SCSE) and Receptive Field Block (RFB) to the proposed network. 
We measure our performance on Cityscapes pixel-level segmentation, and achieve 70.72% class mIoU. We evaluate the trade-offs between mIoU, and number of operations measured by multiply-add (MAdd), as well as the number of parameters.

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
We also include Mixed-scale DenseNet/RF-Mixed-scale DenseNet, ShuffleNetPlus/RFShuffleNetPlus, SE-DPShuffleNet, SE-Wide-ResNetV2 implementation in this repository.  


## Results
Mixed-scale DenseNet/RF-Mixed-scale DenseNet, ShuffleNetPlus/RFShuffleNetPlus, SE-DPShuffleNet, SE-Wide-ResNetV2 under-training (Ask a friend for help)

### Results on Cityscapes (Pixel-level/Semantic Segmentation)

| Model | GFLOPs | Params |gtFine/gtCoarse| mIoU Classes(val./test) | mIoU Cat.(val./test) | Result(*.cvs) | Pytorch Model&Checkpoint |
|---|---|---|---|---|---|---|---|
|MobileNetV2Plus|117.1?|8.3M|Yes/No|73.89/70.72|88.72/87.64|[GoogleDrive](https://drive.google.com/open?id=1b1NJhe4sQ126d7xqg-d9mf8WNTstAoER)|[GoogleDrive](https://drive.google.com/open?id=19s7mdCJqTgZ17hgN7_t17sP-RM_FibmW)|
|RF-MobileNetV2Plus|87.6?|8.6M|Yes/No|72.37/70.68|88.31/88.27|[GoogleDrive](https://drive.google.com/open?id=1JmB5KNmMV92yk5qtjwZnX-ZOhU35Pk6Y)|[GoogleDrive](https://drive.google.com/open?id=1QKLJ7u3DKKOTrMGQCFOprqQZWVrmWQm7)|
|SE-WResNetV2|?|?M|Yes/No|80.13/77.15|90.87/90.59|/|/|


## Contact
ansleliu@gmail.com  
h.liu@tu-braunschweig.de

We are working on the implementation on other frameworks.
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
[18]. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186v6)  