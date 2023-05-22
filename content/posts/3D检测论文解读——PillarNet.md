---
title: "3D检测论文解读——PillarNet"
date: 2023-05-22T15:38:25+08:00
description: "PillarNext前传，以2D稀疏CNN进行pillar特征编码，neck加入特征聚合机制"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 215557.png"
tags: ["next","Pillar","3D","point"]
theme: "light"
---

# 一句话总结

PillarNet是上交马超团队于2022年5月发表的一篇基于pillar的点云3D检测算法，马超团队近年来一直在零星发表3D检测相关文章，比较知名的是LC融合3D检测方法PointAugmenting。

该文章主要创新点是将pillar特征编码有MLP由PointNet方法改为2D稀疏卷积方法，对Neck引入特征聚合机制，对head进行了小改动。

有意思的是：QCraft的杨晓东团队在一年后发表的类似的文章PillarNext，与PillarNet不同的是其Pillar编码使用的是基于稀疏卷积的ResNet18，Neck采用2D检测领域的优秀Neck设计ASPP。相当于都是对pillar特征编码和neck做改动，其中pillar编码都是使用2D稀疏卷积实现，只是具体实现方式不同。

# 摘要

**背景**：速度和精度对3D检测来说都很重要。最近的高精度3D检测器大都基于point或voxel结构，都存在计算消耗大的问题。 与之相对的基于pillar的方法只使用2D卷积网络，速度快很多，但是检测精度也差很多。 

**PillarNet的解决方法**：PillarNet团队仔细研究了pillar方法与voxel方法在检测精度方面的区别，进而提出一个速度快、精度高的3D检测框架PillarNet. PillarNet框架可以划分为3个结构：

- pillar特征编码模块；
- 用于空间语义信息融合的neck；
- 常用的检测head；PillarNet只用2D卷积网络，pillar的尺寸设置灵活。此外，我们还设计了与目标朝向解耦的IOU回归损失机制和IOU-aware的预测机制。

**实验结果**：在Waymo和NuScene上SOTA。代码仓：https://github.com/agent-sgs/PillarNet

# PillarNet整体结构

基于grid的3D检测方法主要分为基于SECOND结构和基于PointPillar结构。其中SECOND结构检测精度高但是计算复杂、PointPillars结构简单但是检测精度低。该文章详细分析了两种架构的异同，进而提出PillarNet结构。PillarNet整体结构如下图所示：![image-20230522153511353](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153511353.png)

## SECOND结构与PointPillars结构对比与分析

**SECOND结构**：SECOND是典型的基于voxel的一阶段检测算法。该算法主要包含三个模块：

- 将非空voxel特征以1、2、4、8的下采样倍率进行3D特征编码；
- 将上一步的特征展平到BEV平面，然后进一步提取特征；
- 基于BEV特征进行检测框和分类的回归；

**PointPillars结构**：该框架将原始点云映射到BEV平面，然后用2D CNN提取特征以及多尺度特征级联。

**分析**：人们认为使用3D稀疏卷积提取的特征比PointPillars的特征表征能力更强、携带信息更丰富。因此基于pillars方法的演进方向主要分两类：要么使用注意力机制进行特征提取、要么使用复杂的多尺度特征聚合方式。这些方法虽然有提升，但是相比SECOND方法仍然差很多，而且速度慢了很多。 我们认为Pillar方法比SECOND方法差的主要原因在于稀疏特征编码学习到的空间特征不够有效、neck模块的空间-语义特征融合不够有效。此外，PointPillars输出的特征图尺寸与点云范围、pillar尺寸有关系，不适用于大维度点云。 本文针对pillars检测框架的问题，提出encoder-neck-head结构的检测网络，提出改进的encoder和neck模块。其中主要改进点为：

- 2D稀疏卷积：在2D平面也使用稀疏卷积提取BEV特征；
- neck语义特征抽象：在低分辨率特征图上进一步进行语义特征抽象；

SECOND、PointPillars、PillarNet结构对比如下图所示：

![image-20230522153536512](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153536512.png)

## PillarNet特征编码模块设计细节

特征编码模块采用4层2D稀疏卷积，下采样倍数分别为1、2、4、8。这种设计可以使用图像领域广泛应用的VGGNet/ResNet等结构，同时下采样设计可以提供不同pillar尺寸的特征。

## Neck设计

FPN结构中的neck模块的作用是：将高维度的语义特征与低维度的空间特征进行融合，进而将融合特征输入給head。 PillarNet基于SECOND的Neck提出两种neck设计，如 下图所示：

![image-20230522153600230](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153600230.png)

# 实验结果

![image-20230522153622932](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153622932.png)

![image-20230522153631911](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153631911.png)

![image-20230522153643079](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153643079.png)

# 个人分析与总结

**作者信息**：该文章是上交马超团队于2022年5月发表的对PointPillar进行优化的点云3D检测框架。马超团队在3D检测领域投入不大，会零星发表几篇相关文章，而且大部分文章都是基于LC融合的方法。相对有名的文章是PointAugmenting和LIFT。我这几篇文章我们可以看出马超团队的风格：通过对已有的简单框架进行简单优化，来尽可能提高检测结果。这种风格正好适合工业场景的落地，我们不妨对此文章保留一点期待。

**创新点**：主要对pillar特征编码和neck做了优化，具体优化如下：

- pillar特征编码由2D稀疏CNN替代；
- neck加入特征聚合机制；

值得注意的是，一年后的2023年5月，QCraft的杨晓东团队提出了PillarNext，优化点与此文章几乎一样：用稀疏2D CNN的ResNet18做pillar做特征编码、用ASPP替代传统neck。如何杨晓东的文章可以称为PillarNext的话，马超团队的这篇文章就足以称为”PillarNext前传“，起名还是保守了，哈哈哈。

**大话演进方向**：上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持辩证的态度阅读。分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。但是后续我会进行独立实验及相关的研究跟踪，预知后续分析请保持关注。

话不多说，上菜：

- 其他2D BackBone:是否可以用其他的2D优秀BackBone来进行pillar特征编码？只要把2D传统CNN改为2D稀疏CNN，这方面可以进行进一步的尝试；

- 其他2D Neck的替换:是否可以用2D检测领域的其他neck替换pillar框架的neck？

- 移植先进训练技术另一个“next”辈的点云3D框架PointNext的一大重要贡献就是将先进的训练技术应用于之前的经典框架PointNet++，从而大大提高了检测精度。pillar系列的next是否也可以用这些先进的训练技术来老树发新芽？

  