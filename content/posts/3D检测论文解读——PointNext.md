---
title: "3D检测论文解读——PointNext"
date: 2023-05-19T17:03:28+08:00
description: "PointNet++老树发新芽，我们且看它能否焕发第二春~"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 212157.png"
tags: ["PointNext","3D","point"]
theme: "light"
---

# 一句话总结

该文章是阿卜杜拉国王科技大学的Bernard Ghanem和钱国成团队提出的，Bernard大佬之前提出过点云3D检测框架PointRGCN。个人认为，该团队对PointNet系列框架的理解是比较深刻的，他们的观点值得我们去认真审视。

这篇文章的主要观点是认为PointNet++的潜力还没被完全挖掘出来，因此从两方面进行优化：

- 应用先进的训练策略，比如AdamW优化器、交叉熵损失函数、数据增强技术等；
- 改进PointNet++框架，提出Inverted Residual MLP和感受野扩张技术；

该框架在语义分割和目标分类数据集上进行了验证，但是从理论上讲，在3D检测数据集上应该也有提升。

总的来说，这篇文章应该称为PointNet plus pro，而不是“next”。因为是基于PointNet++的演进，并没有完全替换或大规模改动PointNet++的结构。此外，该文章对PointNet++各种组件及参数进行了详细的分析，不论对框架的改动是否有效果，这些验证分析工作也是很有价值的、值得我们思考的成果。

anyway, 这篇文章为我们打开了PointNet系列的演进思路，值得我们进一步尝试。

# 摘要

**背景**：PointNet++是一个很有影响力的点云3D检测框架。虽然其检测精度远低于最近的PointMLP、PointTransformer等框架，但是作者团队发现，这些模型的精度提升大都因为训练策略的改进，包括数据增强、优化技术、模型规模等，而不是因为架构上的创新。 

**该团队的研究成果**：该团队认为：PointNet++的潜力还没有被完全挖掘。于是，该团队重新审视了PointNet++网络，系统性研究了模型训练和scaling策略，然后产生两个研究成果：

- 训练策略优化：提出一些列改进的训练策略，大大提高了检测精度，AP能提高9个点左右；
- 提出PointNext模型：基于PointNet++, 提出inverted residual bottleneck和separable MLP设计，使得model scaling更高效更有效；

**实验结果**：

- 在分类数据集ScanObjectNN上SOTA，比PointMLP推理速度快10倍；
- 在语义分割数据集S3DIS上SOTA，精度超过PointTransformer;

代码仓：https://github.com/guochengqian/pointnext

# 基本介绍

PointNet和PointNet++系列是最先直接处理点云的框架，之后演进出KPConv和PointTransformer，并且效果大幅提高。这给人一种错觉，即PointNet系列太简单，不足以学习复杂的点云特征。 该团队重新审视了PointNet++，发现该经典框架的潜力还没被完全挖掘出来，后期文章的提升主要因为训练策略和model scaling策略。已这些优秀的训练策略重新训练PointNet++，提升巨大。具体效果见下图：

![image-20230519170013699](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170013699.png)

# PointNet++ Review

PointNet++整体架构如下图所示：

![image-20230519170033512](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170033512.png)

其中encoder模块由一系列set abstraction模块构成；decoder模块由一系列特征聚合模块组成。 Set Abstration模块包含以下几个组件：

- 下采样层：对输入点云进行下采样；
- grouping层：负责领域点云查询；
- 共享MLP：用于特征提取；
- reduction层：负责领域特征聚合；

# PointNext设计细节

PointNext是基于PointNet++的演进设计，主要创新点包括两方面：训练策略优化和架构优化。

## 训练策略优化

该团队进行了数据增强和优化技术发面的系统研究，进而量化各个技术的效果。

**数据增强**：原始的PointNet++使用的数据增强方法是随机旋转、缩放、变换和抖动。最近有更强的数据增强方法，比如KPConv里的randomly drops colors，Point BERT里的点云重采样策略。PointNext团队对各种数据增强方法进行了专门的量化效果研究。具体实验过程如下：

1. 原始PointNet++为基线：基线版本采用PointNet++原始的数据增强方式；
2. 控制变量验证效果：依次移除每种数据增强方式，验证其是否有效；
3. 去伪存真：移除没用的数据增强方式，保留有用的；
4. 验证其他框架种采用的典型数据增强技术：包括点云重采样、随机噪声、height appending、全场景输入、color drop、color auto contrast等。

实验数据如下：

![image-20230519170114194](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170114194.png)

**优化技术** 优化技术包括损失函数、优化器、学习率调度器(learning rate schedulers)、超参数。 对于上述几个优化技术，PointNet++使用的有：交叉熵损失函数、Adam优化器、exponential learning rate decay。 受益于机器学习理论的发展，现在的神经网络可以用更好的优化器，比如AdamW；还有更高级的损失函数，比如CrossEntropy with label smoothing。 与数据增强的验证方式一样，该团队也对各优化技术进行了控制变量实验，具体流程如下：

1. 超参搜索：查找最优的学习率和weight decay的超参；
2. 研究label smoothing、优化器、学习率调度器的最优组合；

实验结论是：CrossEntropy with label smoothing， AdamW, Cosine Decay在大部分场景下都表现优秀。

## 架构优化

基于PointNet++框架演进，提出PointNext。该框架的主要改动包含两个方面：感受野扩张(receptive field scaling)和模型扩张(model scaling)。

**感受野扩张**：感受野是神经网络中非常重要的一个角色，点云中至少有两种方式可以增加感受野：

- 增加邻域查询半径；
- 采用层级架构：该方式已在原始PointNet++架构中使用；PointNet++的感受野半径设置很大程度上基于直觉，即当对点云进行下采样时，感受野半径翻倍。该团队测试了不同的感受野半径，然后发现不同数据集下最优感受野半径不同，但不论什么数据集，感受野半径对检测精度的影响都很大。具体细节信息如下：![image-20230519170136526](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170136526.png)

此外，该团队还发现邻域查询时，相对坐标距离导致优化困难，进而影响检测指标。因此，该团队对相对位置进行归一化处理，降低了优化难度。 这个问题的主要原因是：weight decay时会降低网络的权重，相当于降低了相对位置的影响。而归一化降级了相对距离的方差，对相对距离的分布进行了重排，进而降低了weight decay的影响。 

【注】感觉这块的逻辑没说清。

**模型扩张**：PointNet++模型很简单，参数量在百万级别，远小于其他模型。该团队发现无论是增加更多SA模块还是增加更多通道，都会导致梯度消失和过拟合，不能提高精度。进一步分析之后，该团队提出Inverted Residual MLP模块，级联在SA模块之后，被证明能有效实现model scaling。InvResMLP是基于SA模块的优化，与SA主要有3个不同点：

- 输入输出见的残差连接：用于减轻梯度消失问题；
- Separable MLP：用于降低计算量、增强点云特征提取，该设计受MobileNet和ASSANet的启发；
- Inverted bottleneck：增加输出通道，提取更丰富的特征；

# 实验结果

在数据集S3DIS/ScanNet/ShapeNetPart上测试语义分割，在ScanObjectNN/ModelNet40上测试目标分类。

**训练细节**：label smoothing CrossEntropy损失函数、AdamW优化器、初始学习率0.001、weight decay 0.0001、batch size为32。训练硬件为V100 32G单卡。验证集上最好的模型用于测试集测试。 具体指标这里就不放了，笔者当前只关注目标检测领域。

**消融实验**：如下表所示：

![image-20230519170200093](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170200093.png)个人分析与总结

**作者信息**：阿卜杜拉国王科技大学的Bernard Ghanem和钱国成团队，值得注意的是PointRGCN也是由Bernard团队提出的，可以说是专注Point框架30年~

**创新点**：包括两方面：

- 以先进的训练技术重新训练PointNet++;
- 基于PointNet++框架做模块级演进；

总的来说，证明了PointNet++仍然很强，值得进一步研究和演进，该框架还没到天花板。

**大话演进方向**：

上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持批判态度阅读。

分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。

但是后续我会进行独立实验及相关的研究跟踪，预知后续发展请保持关注。

话不多说，上菜：

- 适配其他数据增强方式：数据增强已经是检测领域的重要技术，是否有更适合点云的数据增强技术？对点云3D检测任务来说，哪些数据增强方式是有效的。相信像该文章的分析方式针对3D检测任务分析之后，会进一步提高多种检测框架的指标；

- 适配其他先进训练技术：该文章用AdamW替代Adam，是否还有其他优化器的效果能达到更好？2D检测领域使用的先进优化器有哪些，将那些优化器移植到3D检测场景，也许会有进一步的提高；

- 适配其他感受野扩张技术：同是”next“辈框架的PillarNext认为只要简单的下采样就可以提供足够的感受野，而head的上采样又可以恢复出足够的细粒度几何信息。PointNext是否可以用这种感受野扩张技术？是否还有其他有效的感受野扩张技术？经过系统的对比分析后，可能会寻找出更适合点云框架的感受野增强技术；

  