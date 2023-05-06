---
title: "3D检测论文解读——LoGoNet"
date: 2023-03-25T21:39:49+08:00
description: "首次提出局部特征对齐及聚合的理念"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-18 221642.png"
tags: [“local align", 3D", "LC Fusion"]
theme: "light"
---

# 摘要

LC融合方法已经在3D检测领域展现出优秀的效果。

目前基于多模融合的主要融合方法是全局融合，即图像特征和点云特征在全域进行融合。

但是这种全局的融合方法无法在区域级进行融合微调，导致融合效果无法达到最优。

**本文方法**

本文提出一种新颖的局部-全局融合机制：LoGoNet(Local-to-Global fusion network),。

该框架可以在全局和局部两个level进行camera特征和点云特征的融合。

- 全局融合：基于之前已有的方法，只是以点云质心代替voxel特征，从而提高多模特征对齐的精度。
- 局部融合：我们将proposal划分成grid，然后将grid的中心点映射到图像上，之后采样映射点周围的图像特征，将grid中心特征和映射点周围的图像特征进行融合，从而最大化利用proposal周围的纹理特征。
- FDA：之后我们还提出了特征动态聚合模块FDA(Feature Dynamic Aggregation), 该模块用于局部融合特征和全局融合特征之间的交互，进而生成信息更丰富的多模融合特征。

**实验结果** 在Waymo和KITTI 3D上SOTA。 代码仓：https://github.com/PJLab-ADG/LoGoNet



# 基本介绍

3D检测任务的目的是对目标进行分类及3D空间的定位，是自动驾驶任务的关键模块。

- **背景情况**：Lidar和camera是两种广泛应用的传感器，其中Lidar提供准确的深度信息和几何信息，并且已经有很多基于Lidar的3D检测方法取得了不错的效果，比如PointPillars/PV-RCNN/SECOND/CenterPoint/SESSD/VoxelNet等。

  然而，由于lidar的固有特性，导致远处的点云很稀疏，进而影响了检测效果。

  为提高检测精度，人们自然而然想到用图像的语义信息和纹理信息来补充点云信息。

  最近的方法主要是基于全局融合的方法，比如TransFusion/MV3D/Focal Sparse Convolution/EPNet/AVOD/Homogeneous multi-modal fusion/VFF/BEVFusion/PointPainting/PointAugmenting/PI-RCNN/3D-CVF/CAT-DET等。

  

- **已有方法的问题**：上述方法的主要问题是只在全局区域进行LC信息对齐，无法在局部区域进行对齐微调。

  而前景点云数量在每帧点云中的数量占比小于0.1%，所以基于全局区域融合的方式很容易造成关键区域不对齐，进而影响检测精度。

  

- **本文解决方法**：为解决局部区域对齐问题，我们提出LoGoNet检测框架，在局部和全局两个维度进行LC信息对齐。LoGoNet的关键模块有三个：

  1. GoF(Global Fusion): 基于已有方法构建，比如AutoAlignV2/Homogeneous multi-modal fusion/BEVFusion/PointPainting/PointAugmenting, 在全局范围进行LC融合，通过cross-attention和ROI pooling进行LC特征融合，改进点是以voxel内点云质心代替voxel特征，从而提高对齐精度；
  2. LoF(Local Fusion): 目的是在region级信息进行更细粒度的融合，提出PIE(Position Information Encoder)模块，将proposal中的每个grid添加原始点云的位置信息编码，然后将带位置信息的grid中心投影到图像平面，之后使用cross-attention将grid特征与图像平面采样到的图像特征进行融合；
  3. FDA(Feature Dynamic Aggregation): 该模块的作用是将每个proposal的全局特征与局部特征进行融合，通过self-attention生成信息更丰富的多模特征，以供第二阶段的refine模块进行proposal细化。



# 相关工作

- **基于camera的3D检测**：由于camera很便宜，所以很多学者都研究如何基于图像进行3D检测，比如Liga-Stereo/Mono3D++/Smoke/Geometry uncertainty projection network/Pseudo-Lidar++等。由于图像无法直接生成深度信息，因此很多工作都通过图像推导深度信息，然后生成pseudo-Lidar数据，或者将2D特征转换为3D特征，然后在3D空间进行目标检测。最近有一些学者提出基于transformer的检测框架，比如MonoDTR/BEVFormer/PETR/DETR3D。因为很难基于图像推导出精确的深度信息，因此基于图像的3D检测效果比lidar的差很多。
- **基于Lidar的3D检测**：基于lidar的检测方法可以分为3类：
  - point方法：使用MLP直接提取点云特征，比如PointNet/PointNet++/PointRCNN/PointGNN等；
  - voxel方法：将点云转换为voxel，然后使用3D稀疏卷积提取voxel特征，比如MppNet/Voxel RCNN/PDA/Lidar RCNN/Pyramid RCNN/INT/SECOND/VoxelNet等。还有在voxel上用transformer的方法，比如SST/Voxel Set Transformer/VoTr/CT3D等；
  - point-voxel结合方法：将Point BackBone和Voxel BackBone结合，比如Structure Aware Single-Stage/Lidar RCNN/PV-RCNN/STD等。
- **基于多模的3D检测方法**：此类方法是最有潜力的一个方向，因为该方法可以结合Lidar和camera的优点。早期方法有AVOD, MV3D, Frustum PointNet, 在proposal进行LC特征融合；CLOCs是后融合方法，将L和C的检测结果进行融合；PointPainting/PointAugmenting/PI-RCNN用图像的语义信息增强点云特征；3D-CVF和EPNet使用可学习的标定矩阵进行多模特征融合；还有一些方法使用cross-attention进行LC自适应对齐和特征融合，比如AutoAlignV2/DeepFusion/MMF/CAT-DET等。

# 整体结构

- **输入数据**：一帧点云和T个camera产生的图像：
  - 点云数据：$P = \{(x_i, y_i, z_i)|f_i\}_{i=1}^N$，N个点，f为反射强度；
  - 图片数据：$I = \{I_j \in R^{H_I*W_I*3} \}_{j=1}^T$, T个camera；
- **点云处理流程**：采用SECOND和VoxelNet的3D BackBone提取voxel的特征$F_V \in R^{X*Y*Z*C_V}$，其中X/Y/Z为voxel划分grid的尺寸，$C_V$是voxel特征的维度；之后采用SECOND和CenterPoint中RPN生成proposal box $B = \{B_1, B_2, ... ，B_n\}$;
- **图像处理流程**：多个camera的图像数据使用Swin Transformer/Faster RCNN的2D检测器生成稠密的语义特征$F_I \in R^{H_I/4 * W_I/4 *C_I}$;
- **融合处理流程**：使用local-to-global的多模融合模块进行二阶段refine，将点云voxel特征$F_V$、图像特征$F_I$、局部位置信息进行融合； 下面将具体介绍融合模块中的Global Fusion, Local Fusion和Feature Dynamic Aggregation模块。

![image-20230504111226920](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111226920.png)

## GOF(Global Fusion)

**以往的全局融合方法**：之前的全局融合方法往往将voxel质心作为voxel feature的位置，比如Focal Sparse Convolutional Network/AutoAlignV2/EPNet/Homogeneous Multi-modal Feature Fusion/DeepFusion/PointPainting/PointAugmenting/3D-CVF等。 这些方法忽略了每个voxel内点云的分布情况，PDA和Semantic Classification of 3D Point Clouds观察到点云的分布情况更接近voxel的质心。因为point提供了目标的最原始的几何信息，也更容易应对大规模点云。

**本文的方法：CDF——质心动态融合**：针对这种情况，我们设计了CDF(Centroid Dynamic Fusion)，即质心动态融合模块。该模块能在全局voxel空间内，将点云特征和图像特征进行自适应融合。同时，我们使用voxel内的点云质心代表voxel特征的位置。融合过程使用Attention is all you need/Deformable DETR两篇文章提出的deformable cross attention模块，以实现LC特征的自适应融合。示意图如下所示：

![image-20230504111419501](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111419501.png)

其中，$F_V = \{V_i, f_{V_i}\}_{i=1}^{N_V}$ 表示非空voxel特征集合，$N_V$是非空voxel的数量，$f_{V_i}$是每个非空voxel的特征，$V_i$是voxel对应的索引。

基本处理流程如下：

1. **计算voxel质心$c_i$**：voxel特征对应的位置质心为$c_i$, 由voxel内所有point的位置求均值获得；
2. **计算voxel质心映射到图像上的位置$p_i$**：使用映射矩阵$M$将voxel质心映射到图像平面，即$p_i = M c_i$；其中$M$由camera的内参和外参生成。
3. **生成图像聚合特征$\hat{F_I^i}$**：由$p_i$周围的一系列图像特征通过偏移和加权获得。具体计算过程如下：
   + 偏移特征$F_I^k$：偏移是可学习的，计算公式为$F_I^k = F_I(p_i + \Delta{p_{mik}})$;
   + 加权过程：$\hat{F_I^i} = W_m{F_I^k}$;

4. **使用deformabel cross attention生成CDF结果**：![image-20230504111450494](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111450494.png)其中，K是采样的$p_i$周围的图像特征数，M是self-attention head的个数，$A_{mik}$是第k个采样点第m个attention head的attention weight，$\Delta{p_{mik}}$为第k个采样点第m个attention head的采样偏移。

5. **初步生成voxel融合特征**：原始voxel特征$F_V^i$与图像加强的voxel特征级联，得到voxel融合特征$\hat{F_V^*} \in R^{N*2C_V}$；
6. **生成最终voxel融合特征**：使用前馈网络FFN(Feed forward network)减少一半的通道数，获得最终的融合特征$F_V^* \in R^{N*C_V}$；
7. **生成proposal特征**：采用Voxel RCNN和PDA中的方式，对融合特征$F_V^*$进行ROI pooling，生成proposal特征$F_B^g$;

## LoF(Local fusion)

LoF的目的是在融合过程中获得细粒度的几何信息，LoF的关键模块是grid点动态融合GDF(Grid point Dynamic Fusion)，在proposal级进行点云特征和图像特征的动态融合。具体处理过程如下：

1. **将proposal box划分成正方体voxel grid**：将proposal划分成$u*u*u$的等大小的voxel grid，每个voxel grid表示为$G_j$，其中j为grid的索引，每个grid的中心点为$z_j$;
2. **生成grid特征**：使用位置信息编码模块PIE(Position Information Encoder)对每个grid进行位置信息编码，得到$F_G^j$；
3. **grid-ROI特征$F_B^p$** ：使用PIE处理grid特征，获得gird-ROI特征$F_B^p = {F_G^1, F_G^2, ... , F_G^u}$；
4. **计算grid特征$F_G^j$**：使用grid相对proposal质心的相对位置、proposal质心位置、每个grid内point数量进行MLP，获得grid特征$F_G^j$，计算公式为$F_G^j = MLP(\gamma, c_B, log(N_{G_j} + \tau))$, 其中$\gamma = z_j -c_B$，表示grid相对proposal质心的位置，$c_B$为proposal质心位置，$N_{G_j}$表示第j个grid$G_j$中point的数量；
5. **使用GDF(Grid Dynamic Fusion)模块进行grid级的动态融合**：该模块的作用是将图像特征与ROI grid特征$F_B^p$进行动态融合。基本流程是：
   1. grid质心映射：将grid的质心$z_j$映射到图像平面，映射流程与GOF一样；
   2. cross-attention特征融合：使用cross-attention将gird特征$F_B^p$与图像特征融合；
   3. 特征级联：将原始grid特征与图像增强的grid特征进行级联，获得$\hat{F_B^l}$;
   4. 级联特征通道降维：将上述级联的特征降维，获得最终的ROI-grid融合特征$F_B^l$;

LoF的整体结构如下图所示：

![image-20230504111640035](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111640035.png)

## FDA(Feature Dynamic Aggregation)

LoF/GoF、PIE模块之后，我们获得3个特征：$F_B^l, F_B^g, F_B^p$。这3个特征相互独立，没有交互。 因此我们提出特征动态聚合模块FDA(Feature Dynamic Aggregation)，使用self-attention建立不同grid point之间的联系。 具体的处理流程分为两步：

1. **获得融合特征$F_S$**：将三个特征相加获得，即$F_S = F_B^p + F_B^l + F_B^g$；
2. **建立不同grid point间$F_S$的交互**：使用self-attention进行交互；
3. **box refine**：使用FDA生成的flattened feature进行refine；

FDA整体结构如下图所示：

![image-20230504111702200](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111702200.png)

## 损失函数

LoGoNet只训练lidar分支，camera分支不进行训练。

损失函数由RPN损失、置信度损失和回归框损失组成：$L = L_{RPN} + L_{conf} +\alpha L_{reg}$。

其中$\alpha$是超参数，本文设置为1。 整个优化模块参考Voxel RCNN和CenterPoint。

# 实验结果

## 数据集

在WOD和KITTI 3D上测试。两个数据集的基本信息如下：

- **WOD基本信息**：其中WOD是一个很大的数据集，场景分布也很多。包含798个训练序列、202个验证序列、150个测试序列。

  每个序列包含200帧数据，每帧有1分点云数据、5份图像数据。其中点云在x、y平面的扫描范围是[-75.2, 74.2]米，在z轴的扫描范围是[-2, 4]米。

  voxel的尺寸设置为[0.1m, 0.1m, 0.15m]；我们使用AP和APH作为测试指标，以这两个指标在LEVEL1和LEVEL2难度上测试。

  其中LEVEL1之评估点云数超过5个点的目标，LEVEL2则评估点云数超过1个点的目标，WOD上3D检测榜单的排名参考指标是mAPH(L2)。

- **KITTI 3D基本信息**：KITTI 3D数据集包含7481个训练样本和7518个测试样本，测试指标是AP。

  KITTI 3D每帧点云在x方向的扫描范围是[0, 70.4], y方向的扫描范围是[-40, 40], z方向的扫描范围是[-3, 1]。

参考3D object proposals for accurate object class detection对数据集进行划分。 voxel尺寸设置为[0.05, 0.05, 0.1]；

## 参数设置

- attention head数量M：4；
- 采样point数K：4；
- 每个voxel内grid的尺寸u：6；
- 3D backBone: WOD使用CenterPoint的，KITTI 3D使用Voxel RCNN的；
- 图像BackBone: 使用Swin Transformer的Swin-Tiny; + 图像resize: 尺寸缩减一半；
- 图像特征输出通道数：64；
- 数据增强：缩放、翻转、旋转；
- 后处理NMS阈值：WOD上0.7， KITTI上0.55；
- 训练epoch数：CenterPoint 20epoch, Voxel RCNN 80 epochs, LoGoNet 6 epoch;
- BatchSize: CenterPoint 8, Voxel RCNN 2;

## 实验结果

SOTA, 使用TTA时，超越之前SOTA文章BEVFusion 1.05个mAPH(L2), 使用5帧序列输入时，超越MPPNet 16帧序列输入指标1.43个mAPH(L2)。 详细结果如下表：

![image-20230504111727712](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111727712.png)

![image-20230504111737338](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111737338.png)

![image-20230504111748896](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111748896.png)

![image-20230504111758881](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504111758881.png)

# 个人分析与总结

**作者信息**：这篇文章是上海AI实验室的乔宇、YikangLi 团队于2023年3月提出的一篇基于LC融合的3D检测算法，在KITTI和WOD上的排名都很靠前，刚发表时甚至登顶榜首。

**创新点**：该方法的主要创新点在于：提出局部融合的概念。

理由是：之前的方法都是基于全局融合，然而前景点(目标上的反射点)数量在所有电云中的比例小于0.1%。如果基于全局融合，势必导致局部特征对齐不准确，会为追求全局对齐的精度而牺牲局部目标特征的对齐精度。

**其他组件**：其他组间都是使用已有方法：

- 点云3d 特征/2d 特征使用second的结构；
- 图像特征使用Swin Transformer的结构；
- 图像特征在全局和proposal级的融合使用transformer的cross-attention和self-attention, 与M3DETR/CenterFormer/CT3D的方法大同小异；

**大话演进方向**：上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持批判态度阅读。

分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。

但是后续我会进行独立实验及相关的研究跟踪，预知后续发展请保持关注。

话不多说，上菜：

- **去除全局融合模块**：总的来说，局部特征融合是有效的。通过消融实验可以看出，局部融合能提高1-2个mAPH的百分点。

  但是局部融合和全局融合可能存在特征重复，如果进行进一步演进的话，可以考虑去除全局融合模块。

  因为从理论上讲，如果proposal级进行过LC特征融合，那么基于这些特征的RPN和Refine足够进行目标检测，没必要再生成携带大量背景特征的全局融合特征。

  此外，从消融实验也可以看出，在局部融合的基础上加上全局融合，提升的mAPH只有0.1-0.5个百分点，带来的增益很小。

- **与其他方法结合**：LoGoNet的3d BackBone可以尝试使用更先进的，比如DSVT; 此外可以尝试点云补齐方法，比如GLENet; 也可以与多帧序列方法结合，比如MPPNet。

- **尝试新的融合方法**：目前该文章使用self-attention和cross-attention机制进行LC特征融合，接下来可以探索是否有更有效的融合方式。