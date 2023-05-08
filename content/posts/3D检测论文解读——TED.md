---
title: "3D检测论文解读——TED"
date: 2023-05-08T13:24:27+08:00
description: "TTA+model ensemble的数据增强前移，以一个模型对多个增强数据进行特征提取和聚合，并对特征提取和聚合模块做了相应的适配"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-18 175341.png"
tags: ["数据增强","3D","LiDAR"]
theme: "light"
---

# 一句话总结

厦门大学王程团队与赢彻科技杨睿刚团队合作发表于2022.11，主要创新点是将TTA的数据增强机制前移，并对特征提取和特征聚合模块做了相应适配，以单模型代替传统的TTA+model  ensemble机制的多模型推理。

![head](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/head.png)

实验结果还行，在KITTI 验证集上对car moderate的检测精度达85.28, WOD上没什么竞争力。3090上推理速度11fps，显存消耗翻倍。

![屏幕截图 2022-10-18 175341](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-18 175341.png)

# 摘要

**背景**：3D检测对自动驾驶很重要，之前的检测方法没有考虑航向变化和放射变换(variations of rotation and reflection transformations)，因此需要大网络和数据增强来提高鲁棒性。 最近的**equivariant network**对点云进行多重变换，然后使用共享网络来对变换进行建模，显示出对目标几何结构建模的强大潜力。 但是Equivariant network很难直接应用到点云3D检测任务上，因为其计算量很大，推理速度很慢。

**本文方法**：本文提出TED(Transformation Equivariant 3D Detector), 解决了Equivariant在点云上的应用问题。基本处理流程是：

1. 使用稀疏卷积3D BackBone提取多通道等价变换(multi-channel transformation-equivariant)的voxel特征；
2. 进行特征对齐和特征聚合，将上述的等价特征转换为轻量化的表征形式；

**实验结果**：在KITTI 3D上排名第一。 代码仓：未开源。



# 基本介绍

当目标的朝向变换时，我们希望检测框的朝向也跟着变换，但是其他参数都不变。但是目前的模型大都无法解决此问题。 本文的TED的目标就是解决此问题。

**【注意】**这是一篇数据增强方法吗？

最近的equivariant network通过共享卷积网络解决了变换问题，但是计算量太大，无法实时运行。



本文的TED解决了计算量问题。关键模块有3个：

- TeSpConv(Transformation-equivariant Sparse Convolution) BackBone: 使用共享权重来记录等价变换的voxel特征；
- TeBEV(Transformation-equivariant Bird Eye View) pooling: 在场景级进行等价变换特征的轻量化转换；
- TiVoxel(Transformation-invariant Voxel) pooling: ：在目标级进行等价变换特征的轻量化转换；
- DA-Aug(Distance-Aware data Augmentation): 增强稀疏目标的几何特征；

# 相关工作

最早的Lidar3D检测方法是将3D形式的点云转换到BEV平面再检测；最近的方法大都基于voxel和point，比如SECOND/PointPIllars/SA-SSD/SE-SSD/Voxel RCNN/SFD/3DSSD/SASA/PointRCNN/PV-RCNN/STD/CT3D等。 本文的TED框架以2阶段检测的基于voxel的pipeline为基础，在等价变换方面进行了扩展。

**等价变换模型**：Transformation equivariance，比如

- Spherical CNNs(2018):
- General E2 Equivariant Steerable CNNs(2019):
- Group Equivariant Convolutional Networks(2016):
- Learning Stterable Filters for Rotation Equivariant CNNs(2018):还有一些基于voxel和point的处理3D数据的等价变换模型，比如：
- SO3: Vector Neurons: A General Framework for SO3 Equivariant Networks(2021);
- Equivariant Point(2021):
- 3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data(2018);还有专门用于目标检测的模型，比如：
- ReDet: A Rotation equivariant Detector for Aerial Object Detection(2021);
- Body Constitution and Unhealthy Lifestyles in a primary(2021);这些方法都太复杂，消耗太多计算量，本文TED针对这些方法做了实时性运行的优化。

**等价变换的及其不变性**：将X空间变换为Y空间的等价变换公式为$f[T_g^X(x)] = T_g^Y[f(x)], x \in X, g \in G$. 其中T表示变换操作。



# TED具体结构

主要处理流程是使用TeSpConv提取并缓存多通道变换的voxel特征；使用TeBEV和TiVoxel来对齐和聚合等价变换的特征。整体结构如下图所示：

![image-20230508132106262](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132106262.png)

## Transformation Equivariant Voxel BackBone

为了从原始点云中高效提取等价变换特征，我们设计了Transformation equivariant sparse convolution 3D BackBone--TeSpConv.该BackBone基于稀疏卷积，但是稀疏卷积对旋转和反射不等价。 为解决稀疏卷积在旋转和反射方面的等价问题，我们增加了变换通道。通过以下两种方式我们实现了等价：

- 对输入点云进行不同的旋转和反射变换；
- 提取每种变换的voxel特征，变换通道间权重高度共享；

## Transformation Equivariant BEV Pooling

1. 将上一步的每种变换看作一个通道，共有2N个变换，因此有2N个特征通道。
2. 然后将所有voxel特征在高度方向上压缩，获得BEV特征。
3. 将3D空间的grid point映射到BEV上，然后用双线性插值获得2N个聚合特征。
4. 对上述的2N个聚合特征进行max-pooling, 获得最终的BEV特征。

示意图如下所示：

<img src="https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132129663.png" alt="image-20230508132129663" style="zoom:150%;" />

## Transformation Invariant Voxel Pooling

该模块的作用是在在目标级进行多变换通道的特征聚合。具体做法是：

1. 将原始3D空间的proposal转换到各个变换的空间，获得对应的voxel特征；
2. 使用VSA(Voxel Set Abstration)对多个变换的voxel特征进行聚合；
3. 使用cross-grid attention对第2步获得的多个grid特征进行进一步聚合，以获得更好的几何特征。

## Distance Aware Data Augmentation

远距离目标的几何信息缺失往往会导致检测指标大幅度下降。 为解决此问题，我们从近处的稠密目标生成稀疏训练样本，来提高几何提取能力。 最简单的采样方法就是随机采样或FPS(Farthest Point Sampling)，但是这种方法会破坏点云的分布形式。

为解决此问题，我们提出了distance-aware采样策略。具体机制是：

1. 对于gt box及其内部point，对其位置加随机偏移$\alpha$;
2. 将点云及voxel转换为球形；
3. 将每个voxel内最靠近球心的点作为采样点；
4. 为模拟遮挡情况，对上述的采样点进行一定比例的随机丢弃；

# 实验结果

**实现细节**：

- 版本：我们实现了基于单模Lidar的TED-S版本和基于多模的TED-M版本。
- 变换方式：对于变换方式，采用3种旋转变换方式和2种反射变换方式。
- proposal数和NMS阈值:在KITTI数据集上，proposal个数和NMS阈值与Voxel RCNN保持一致；在Waymo数据集上与PV-RCNN++保持一致。
- 训练硬件：2块3090；
- batch size: 4;
- 学习率：0.01；

**实验结果**：具体信息如下表：

![image-20230508132206414](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132206414.png)

![image-20230508132216101](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132216101.png)

![image-20230508132228366](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132228366.png)

总的来说：

- KITTI 3D上优秀：在KITTI 3D上测试集上，TED-M对car-moderate样本的检测率到达85.28%, 介于用TTA和没用TTA方法之间，算是很优秀的指标，但是没公布TED-S的，估计没什么竞争力；
- WOD上没竞争力：在WOD验证集上对vehicle L1样本的mAP为79.26，远低于同时期的SOTA方法；

**消融实验**：

- 旋转变换次数：在3和4之间达到最高检测率，为实时性，将变换次数设为3，可以在3090上达到11fps的帧率；
- 各个组件有效性分析：具体信息如下表：![image-20230508132251246](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230508132251246.png)

**不足点**

- 与Voxel RCNN相比，消耗的GPU内存翻倍；
- 数据变换时没有用缩放，该方法在实践中应用较少；
- 因为变换过程涉及voxel，所以不是完全的等效变换，voxel尺寸越小越接近等效变换，但是计算量越大；



# 个人分析与总结

**作者信息**：厦门大学的王程团队与赢彻科技的杨睿刚团队合作，发表于2022年11月。

值得注意的是，王程团队与MPI的史少帅与次年3月发表了VirConv，同样在榜单上排名很靠前。只不过两篇文章用的完全不一样的方法，这篇TED应该是与后来的VirConv是同时期进行的工作，没有演进的关系。



**创新点**：核心思路是数据增强前移，对输入数据进行各种变换，然后用同一个模型进行特征提取与聚合，其中特征提取和聚合做了相应的适配修改。与TTA+model ensemble方法相比，速度节省了很多，数据增强的效果差不多。

在结果方面，与主流方法TTA相比，指标稍低一点，但是比未加TTA的方法的指标高。



**大话演进方向：**上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持批判态度阅读。

分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。

但是后续我会进行独立实验及相关的研究跟踪，预知后续发展请保持关注。

话不多说，上菜：

- 特征提取和聚合方式的改进：TED在3D卷积部分对所有变换使用共享权重，这点与多模型TTA+ensemble机制等效。但是BEV特征聚合时候，只是简单使用双线性插值和max-pooling方式聚合多种变换的特征，这种方式过于简单，效果也有限。此外，TiVoxelPooling同时使用了VSA和cross-attention，可能过于复杂，后续可以尝试寻求更简单有效的目标级特征聚合方式。
- 对数据增强方式进行演进：当前只用了旋转、反射和近距离目标点云下采样，后续可以考虑其他增强方式。
- 将TED的前增强机制应用于其他检测框架：将此数据增强方式应用于LoGoNet那样的局部特征对齐框架，实现局部+等效变化结合的特征对齐和特征聚合方式。或者应用于SFD/VirConv/GLENet那样的虚拟点云生成框架，通过虚拟点云+数据增强提高检测模型的鲁棒性和检测精度。或者将前增强机制与MPPNet那样的序列检测机制结合，以增强数据+序列数据来提取更丰富的特征，从而提高检测指标。