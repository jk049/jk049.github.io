---
title: "3D检测论文解读——SFD"
date: 2023-05-06T16:49:18+08:00
description: "第一篇使用图片生成虚拟点云，然后进行LC融合3D检测的文章"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-18 220955.png"
tags: ["虚拟点云","3D","LC Fusion"]
theme: "light"
---

# 摘要

**背景**：当前基于Lidar的3D检测算法不可避免会收到点云稀疏性的影响，很多学者尝试通过多模融合的方法解决此问题。 然而当前融合方法也不成熟，效果也有限。

**本文解决方法**：本文提出一种新的多模融合框架SFD(Sparse Fuse Dense)，该框架的核心理念是生成虚拟点云来解决点云稀疏性问题。关键模块如下：

- 3D-GAF: 3D Grid-wise Attentive Fusion, 一个RoI融合模块，在grid级通过注意力机制将原始点云与虚拟点云融合，获得RoI特征。该融合模块通过局部的细粒度融合，可以获得更高精度的融合特征。
- SynAugment: Synchronized Augmentation，一种多模框架下的数据增强方法。
- CPConv: Color Point Convolution, 针对虚拟点云的特征提取模块，能同时提取2D图像特征和3D几何特征。

**实验结果**：刚发布时，在KITTI 3D上对car的检测指标排名第一。 代码仓：https://github.com/LittlePey/SFD



# 基本介绍

**基于Lidar的3D检测方法**：由于自动驾驶的需求驱动，近几年3D检测算法发展很快。 22年以前主要的3D检测算法大都基于Lidar，比如To the Point(2021)/Fast Point RCNN(2019)/Real-Time Anchor-Free Single-State 3D(2021)/TANet(2018)/Pyramid RCNN/PV-RCNN/PointRCNN/STD/HVNet/SE-SSD等。 上述这些方法的主要缺陷是：点云稀疏性严重影响其检测精度。

**基于多模的3D检测方法**：由于lidar的固有缺陷，学者们提出多模融合的方法来解决此问题。比如MV3D/AVOD/MMF等。 这些方法有两个缺点：

- 融合机制粗糙：2D图像特征直接与lidar BEV特征融合，没有局部关联，而且图像特征包含背景等非目标特征；
- 数据增强不足：很多多模融合方法都使用到了数据增强， 但是，但是基于LC融合的数据增强很难，因为图像数据与点云数据格式差异很大。

**本文解决方法**：

- 3D-GAF：3D Grid-wise Attentive Fusion, 在局部将2D和3D特征进行融合，获得ROI特征。
- SynAugment: Synchronized Augmentation, 解决LC融合领域的数据增强问题。我们通过图像生成虚拟点云，然后直接基于虚拟点云提取图像特征，数据增强也都基于3D空间。
- CPConv: Color Point Convolution, 用于从虚拟点云中提取图像特征，能同时提取图像特征和几何特征。具体做法是将虚拟点云投射到图像平面，然后使用RoI领域搜索机制，在局部进行特征提取。

# 相关工作

**基于lidar单模的3D检测**：SECOND、SA-SSD、PV-RCNN、Voxel RCNN、SE-SSD、CenterPoint、Lidar RCNN、SPG、VoTr、Pyramid RCNN、CT3D等。这些方法都是在点云稀疏的前提下尽量提高检测精度。

**基于多模的3D检测**：学者们希望借助其他传感器的信息来弥补点云的稀疏性先天缺陷，比如Frustum PointNets、Frustum ConvNet、PointFusion、MV3D、ContFuse、MMF、VMVS、3D-CVF、CLOCS、EPNet、MVX-Net、PointPainting、PI-RCNN等。

**深度补全方法**：深度补全是基于图像来生成一个稠密的深度地图(dense depth map)，相关的方法有：DenseLIDAR、Sparse Auxiliary Networks、PeNet、Depth Completion with Twin Surface Extrapolation。但是这些方法主要用于基于图像的3D检测，很少应用于LC融合的3D检测领域。



# SFD整体结构

将原始点云R映射到图像平面上生成稀疏深度图S(sparse depth map), 将S与图像输入到深度补全网络，生成稠密深度图D(dense depth map)。 从图像生成虚拟点云P(Pseudo clouds)。 SFD整体结构如下图所示：

![image-20230506163714600](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230506163714600.png)

主要包含3个部分：

- 原始点云处理流：原始点云经3D BackBone、RPN生成proposal；
- 虚拟点云处理流：图像、稀疏深度图经深度补全网络生成稠密虚拟嗲云，然后用CPConv提取图像特征和几何特征；
- 原始点云特征与虚拟点云特征融合模块：使用注意力机制，在proposal级进行原始点云特征和虚拟点云特征的融合。

## 3D-GAF

之前的MV3D、AVOD、MMF提取BEV特征和图像特征，然后进行简单的特征级联实现LC融合，这种融合策略不够精细，效果有限。本文通过将图像转为虚拟点云，从而可以进行更细粒度的LC特征融合。示意图如下所示：

![image-20230506163821553](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230506163821553.png)

有了原始点云grid和虚拟点云grid的对应关系之后，我们借鉴EPNet/Squeeze and Excitation Networks/Selective Kernel Networks的思路，使用注意力机制进行两种特征的融合。具体流程如下：

1. proposal voxel划分为$6*6*6$ 的grid，原始grid的特征为$F^r$, 虚拟grid的特征为$F^p$；
2. 将原始grid特征与虚拟grid特征级联；
3. 将级联特征输入全连接层和sigmoid激活层，得到原始特征的权重系数$w^r$和虚拟特征的权重系数$w^p$;
4. 按权重系数，获得融合特征；

公式如下：![image-20230506163840590](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230506163840590.png)

## SynAugmentation

该模块的主要功能是数据增强，主要包含两个部分：

- 按点云处理图像：针对LC融合的数据增强方法面对最大的问题是如何按照点云的形式处理图像。目前的解决方法是深度补全，通过深度补全算法，可以将图像转换为虚拟点云，该虚拟点云携带所有图像信息，之后就可以按照原始点云的方式进行数据增强。
- 3D空间提取图像特征：常用的3D特征提取方法是3D系数卷积，但是还有更有效的方法。本文提出CPConv(Color Point Convolution)来提取特征。



【问】这也没说数据增强呀。。。

## CPConv

虚拟点云由图像生成，因此每个point都与每个像素有对应关系，所以我们可以把虚拟点云表示为$(x_i, y_i, z_i, r_i, g_i, b_i, u_i, v_i)$。 最朴素的提取虚拟点云特征的方式是voxelization之后使用3D稀疏卷积，这这种方式不能充分利用虚拟点云的语义信息和结构信息。 因此我们提出CPConv来进行特征提取。具体流程是：

1. 将每个虚拟点云映射到图像平面；
2. 虚拟点云特征表示为$f_i = (x_i, y_i, z_i, r_i, g_i, b_i)$;
3. $f_i$输入全连接层，减小复杂度；
4. 领域特征表示：搜寻虚拟点云的每个邻域点，虚拟点与第k个领域点的位置残差表示为：$h_i^k = (x_i - x_i^k, y_i - y_i^k, z_i - z_i^k, u_i - u_i^k, v_i - v_i^k, d_i^k)$, 其中d为两个点的位置在三维空间的L2范数。
5. 将位置残差输入全连接层，改变维度；
6. 上述位置残差作为权重参数，与对应的特征进行加权，获得聚合特征；
7. 聚合特征输入全连接层；
8. 堆叠三层CPConv，提取更高维度的特征，将各层CPConv的特征级联起来；

整体结构如下图所示：

![image-20230506163903534](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230506163903534.png)

## 损失函数

$L = L_{rpn} + L_{roi} + \lambda_1 L_{ori_roi_head} + \lambda_2 L_{pseu_roi_head} + \beta L_{depth}$

其中参数值如下$\lambda_1 = 0.5$

- $\lambda_2 = 0.5$
- $\beta = 1$

# 实验结果

在KITTI上测试，SOTA。WOD和NuScene没有标注深度信息，因此没在这两个数据集上测试。

**实现细节**：lidar分支基于Voxel RCNN，深度补全基于PENet，数据增强基于Voxel RCNN和CIA-SSD的方法。

详细数据如下表：

![image-20230506163924974](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230506163924974.png)

**推理速度**：2080Ti上10fps。



# 个人分析与总结

**作者信息**：这篇文章是浙大的蔡登团队提出的一篇LC融合3D检测方法，发表于2022.3.9。蔡登团队之前也发表过3D检测的优秀文章，比如Graph RCNN，PI-RCNN。

**创新点**：该方法首次将虚拟点云生成技术应用于LC融合3D检测领域，对中等和困难样本的检测精度有明显提高。除了对指标有明显提高之外，本文对LC融合3D检测技术也有很大的启发：即通过图像生成稠密的虚拟点云，然后将虚拟点云、图像、原始点云结合进行3D检测。通过这种方式，可以大大降低点云稀疏性对检测精度的制约。

事实上，2023年厦门大学的王程团队和MPI的史少帅提出一篇VirConv，就是基于SFD做的演进：通过图像生成虚拟点云之后删除冗余点云，然后进行特征聚合。VirConv的创新点简单但有效，目前在KITTI 3D上排名第一(2023.5)。

**大话演进方向**：

上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持批判态度阅读。

分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。

但是后续我会进行独立实验及相关的研究跟踪，预知后续发展请保持关注。

话不多说，上菜：

- 虚拟点云生成技术：无论SFD还是VirConv，使用的虚拟点云生成技术都是PENet的方法，该方法最初被用于基于图像的3D检测领域。由于图像的3D检测精度远低于Lidar，因此在学术界在这方面的投入相对较低。目前基于图像虚拟点云增强原始点云的方法已证明可行且有效，而且后续的发展潜力巨大。因此基于图像生成虚拟点云的技术可能或受到更多关注，届时我们可以用更有效的算法生成虚拟点云，进而进一步提高3D检测精度。
- 特征聚合方式：目前广泛应用的特征聚合是transformer的注意力机制，然而当前学术界对transformer的理解还不够深刻。后续随着transformer技术的演进，相信会有更成熟更有效的特征聚合技术涌现出来。我们可以跟踪着transformer的演进，尝试用更有效的特征聚合机制提高3D检测精度。