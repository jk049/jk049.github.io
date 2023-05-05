---
title: '基于LC融合的3D检测算法调研'
date: '2022-12-07T13:06:38+08:00'
description: '盘点近5年的LC融合3D检测典型算法'
author: 'jk049'
cover: 'https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 220855.png'
tags: [”review","LC Fusion","3D"]
theme: 'light'

---

# 基于LC融合的3D检测算法简介

自动驾驶技术已经广泛应用与自动驾驶卡车、机器人出租车、送货机器人等领域。
感知系统的输入通常是多模数据，包括来自camera的图像、来自lidar的点云、以及高精度地图；感知系统的输出是路上关键元素的语义信息和几何信息。
高精度的感知结果是下一步轨迹预测及路径规划的保障。



自动驾驶汽车需要很多传感器来感知周围的信息，比如waymo的自动驾驶汽车装备了29个camera，6个radar，5个lidar。

不同传感器提出不同的信息，因此需要将多传感器信息融合。

lidar和camera是3D目标检测领域中两个重要的可以互补的传感器：

+ lidar点云带有低分辨率的形状和深度信息；
+ camera图像带有高分辨率的形状和纹理信息；

因此，研究人员对基于LC融合的3D检测投入极大热情，想通过两类传感器的特性互补来得到好的检测效果。



## Lidar及camera的数据形式和特点

通常可用于3D目标检测的传感器有：

1. camera：

   优点：

   + 便宜
   + 易获取

   缺点：

   + 只能获取目标的外观信息，无法获取其结构信息；
   + 无法获取准确的深度信息；
   + 易受天气和时间影响；

2. lidar:

   优点：

   + 可获取精密的3D结构信息：一个m束激光的lidar，在一次扫描周期内进行n次测量，可获得尺度为$R^{m*n*3}$的range图像，其中range图像的每个像素有3个通道(range $r$, azimuth $\alpha$, inclination $\phi$)。 range图像是lidar获取的原始数据，可通过将球坐标系转换为笛卡尔坐标系来将其进一步转换为点云数据。点云数据表示为$R^{N*3}$，其中N表示有N个点云，每个点云有xyz轴的坐标值。
   + 不易受天气和时间因素影响；

   缺点：

   + 比camera贵很多；

camera, lidar的相关数据形式如下：
![image-20230108090225226](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108090225226.png)



## LC融合方法分类

然而，在研究早期，基于LC融合的3D检测效果都差于基于单Lidar的检测指标。主要原因就是当时的融合技术还不成熟，而异构传感器融合涉及的数据对齐问题是非常困难的问题，早期的融合技术无法有效解决异构数据对齐问题。

随着近年来LC融合算法的演进，LC融合的指标与单Lidar检测指标之间的gap变得越来越小。直到2022年，基于LC融合的3D检测指标开始反超基于单Lidar的检测指标。

从融合的发生阶段来看，基于LC融合的3D检测算法可以分为3类：

+ 前融合：针对像素或点云的融合方法，对传感器原始数据做修改；
+ 特征级融合：对图像特征和点云特征进行各种各样的融合；
+ 后融合：对图像的检测结果和点云的检测结果进行融合，最后输出融合的检测结果；

其中前融合与后融合属于早期融合技术，不能将LC间的数据进行充分融合，检测效果一般，远不及单Lidar的检测效果。特征级融合算法是主流的融合方式，学术界提出很多有效的方法，目前效果最好的融合算法也是特征级的融合算法。

接下来本文将分别介绍这三种融合方式的典型算法。



## 本文结构

本文第一部分初步介绍了基于LC融合3D检测的背景，包括应用领域、Lidar和camera的特点及融合方法分类；

第二部分介绍了多个LC融合的代表性方法，提炼各方法的核心思想和关键模块，以及该方法在开源数据集上的评价指标；

第三部分对LC融合3D检测领域进行了全面细致的总结，包括3D与2D检测的差异、不同场景下3D检测的侧重点、数据集、评价指标、基于lidar的3D检测方法、基于LC融合的的3D检测方法等内容。



# LC融合代表性方法



## Frustum PointNets for 3D Object Detection from RGB-D Data

【方法分类】前融合，用图像proposal增强点云原始数据

之前的LC融合算法往往是基于voxel或图像的，本文提出一种图像和点云的前融合方法，结果SOTA。
我们提出的方法如下：

1. 以2D检测器在图像上生成2D proposal；
2. 将2D proposal 映射到点云上，形成3D frustum;
3. 使用frustum PointNet在frustum中预测3D框，预测过程分为语义分割和回归两个步骤；

本文提出的检测方法示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1sroNnGbsqP)



本方法的架构图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1srxpe5vFjM)



实验结果：KITTI上SOTA。

![image-20230117224501904](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230117224501904.png)





## PointPainting: Sequential Fusion for 3D Object Detection

本文提出PointPainting方法，将点云映射到图像上，然后对图像进行语义分割，每个point都打一个语义标签，然后将带有语义标签的点云送到lidar专用网络中。

本文提出的方法：

1. 对图像进行语义分割；
2. 将点云映射到语义分割图像上；
3. 用lidar检测模型对语义点云进行3D目标检测；

PointPainting基本流程示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1sxm26cZ7WS)

PointPainting的特点是：

+ 不受feature blurring影响；
+ 对检测架构没约束；
+ 不需要pseudo-lidar；
+ 不限制最大召回率；

实验结果：比Point-RCNN, VoxelNet, PointPillars效果都好。
效果对比如下：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1ssDxZRclOM)

![image-20230117224056215](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230117224056215.png)



## Multi-Task Multi-Sensor Fusion for 3D Object Detection

【方法分类】特征级融合，基于ContFuse的演进

提出一种端到端的模型，通过多个level的融合取得不错效果，在KITTI数据集中的2D, 3D以及鸟视图目标检测任务benchmark。

MMF架构示意图如下所示：![image-20230108093042789](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108093042789.png)
具体处理流程如下：

1. 基于point和ROI进行LC特征融合: 一路提取图像特征，并用FPN将多尺度的图像特征融合起来，一路用ResNet18+FPN提取lidar BEV特征；然后将图像特征融合到BEV特征，特征融合借鉴contFuse方法；之后通过BEV特征检测每个BEV voxel内的3D框；最后做ROI refine获得准确的2D框和3D框；LC特征融合示意图如下所示：

![image-20230118102042062](http://image.huawei.com/tiny-lts/v1/images/c3dd3cdfb22275561b82ea3b55347c1b_569x394.png)

1. 背景估计模块推理道路的几何形状：使用UNet在BEV上回归背景高度；
2. 使用深度补全学习模块获取更稠密的point特征；
3. 推理3D框；



指标提升巨大，速度为10帧每秒。![image-20230108100818398](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108100818398.png)





## Multi-View 3D Object Detection Network for Autonomous Driving

【方法分类】基于proposal的融合，

本文提出一种高精度的基于LC融合的3D检测框架MV3D，具体做法是 将稀疏点云编码为紧凑的多view表示。
MV3D框架分为两个子框架：

1. 3D proposal生成：借鉴RPN的思路，从点云BEV生成3D候选框；
2. 多view特征融合：设计专门的融合机制，将多view的区域特征进行融合；

MV3D框架示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1sy0x7TtVEe)



**一. 点云的表示形式**：

1. BEV表示：使用BEV视图，计算height、density、intensity，然后将三个值归一化：

   + height：将3D点云划分为分辨率为0.1m的2D grid，每个grid的高度特征为grid内最高的点云高度，同时将点云纵向切片m份，每份独立计算高度特征；
   + density：以BEV计算一份密度特征，每个grid中point的数量；
   + intensity：grid中最大高度的point的反射强度值；

   BEV下的特征计算示意图如下所示：
   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1syMPUheBG5)

2. FV表示：作为BEV的补充，BEV太稀疏，FV稠密很多；以height、distance、intensity表示FV，示意图如下所示：

   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1syOqZUuAfh)

**二. 3D proposal 网络**

借鉴RPN(Region Proposal Network，Faster R-CNN)思路，以BEV作为输入， 从一系列3D先验框中生成3D proposal。

**三. 基于区域的融合网络**

1. 借鉴Fast R-CNN中ROI pooling思想，将多个view的特征向量进行融合；
2. Deep fusion：示意图如下所示：
   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1syUwq5gJyj)
3. 3D框回归；
4. 网络正则化：采用drop-path和辅助损失函数；

**四 backbone架构**：

以VGG net为基础，做了如下改动;

+ channel数减半；
+ 针对小目标，插入多个上采样层；
+ 移除第4个池化操作；
+ 增加全连接层；



实验结果：大幅提高检测精度，比当前SOTA的AP提高25%-30%；比单lidar的SOTA的AP高10%。

![image-20230118132642933](http://image.huawei.com/tiny-lts/v1/images/10524929a52781d74a528f71405eae9e_1107x250.png)



## Joint 3D Proposal Generation and Object Detection from View Aggregation

【方法分类】基于proposal的特征级融合+后融合

本文提出AVOD(Aggregate View Object Detection)来进行自动驾驶场景的3D目标检测。AVOD使用两个子网络进行LC特征融合：

+ RPN: 生成3D框proposal；
+ 检测网络：回归生成最终的3D框；

整体架构图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1syvcp14ZrE)

本文主要贡献如下：

+ 基于FPN(feature pyramid networks)提出一种新的特征提取器，可以从点云和RGB图像产生高精度特征图(feature map), 对小目标检测有好处；
+ 提出特征融合RPN, 产生高召回率的region  proposal；
+ 提出一种3D框编码方法，可以提高定位精度；
+ RPN采用1*1卷积核，提高计算速度，降低内存消耗；

实验结果：实时、低内存在KITTI 3D数据集上SOTA. 
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1syamHwC1ej)
代码路径：[GitHub -AVOD](https://github.com/kujason/avod)





## PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation

【方法分类】特征级融合

本文提出基于LC融合的3D检测方法：PointFusion。

PointFusion基本流程如下：

1. 图像由CNN框架处理，lidar点云由PointNet框架处理；
2. 由融合框架将图像和点云的处理结果进行融合；
3. 由3D点云提供anchor，融合后得出3D框的预测位置及置信度；


之前的融合方法大都需要将点云映射到图像，或将点云进行grid划分，不论何种处理方法，都涉及信息损失，并且需要特殊模型来处理点云的稀疏性。
我们的方法保持点云原始表示形式，并且适用异构网络架构。特别地，针对点云数据，我们使用基于PointNet的方法对原始点云进行处理。

PointFusion主要包含3个组件：

1. CNN网络从图像中提取外观和几何特征；
2. 基于PointNet的衍生模型处理原始点云数据；
3. 融合子网络将点云处理结果和图像处理结果进行融合，进而预测3D框。具体预测过程是先在点云上预测box的拐角，然后计算置信度。预测拐角点的思想来源于Faster-RCNN的spatial anchors. 

PointFusion基本处理流程如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tC9MKner7g)



PointFusion 实现细节

1. 点云特征提取网络：基于PointNet框架，移除全连接层之后的batch normalization, 并且对输入进行归一化处理；
2. 融合网络：包括两个部分：
   + 全局融合网络：直接从图像特征和点云特征中回归3D框的8个顶点；
   + 稠密融合网络(Dense fusion network): 将输入点云作为稠密空间anchor，通过offset的方式计算3D框；

实验结果：在KITTI数据集上SOTA。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tCJUhD132H)



## Deep Continuous Fusion for Multi-Sensor 3D Object Detection

【方法分类】特征级融合

使用连续卷积将图像特征和点云特征进行多维度融合。

本文提出的融合方法将图像特征映射到lidar BEV空间，从而实现LC融合。在映射的过程中，我们提出一种端到端的学习网络，采用连续卷积将图像特征和lidar特征进行多维度融合。

我们使用卷积网络提取图像特征，然后将图像特征映射到BEV，最后以卷积层进行特征融合。本文的核心点是映射后的特征融合，使用连续卷积提取相应的特征，本文基本架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tQ5dZl43kX)

1. 连续卷积：连续卷积是作用于非网格-结构化数据(non-grid-structured)上的算子，核心idea是使用多层感知机作为参数化核函数进行连续卷积；示意图如下所示：
   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tRLIxniPkf)
2. 目标检测网络：



测试结果：本方法在KITTI数据集上SOTA，显著提高了各项指标。

![image-20230117193328346](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230117193328346.png)





## EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection

【方法分类】特征级融合

本文使用图像语义特征来增强点云特征，然后用端到端的EPNet进行3D检测。

本文关键贡献：

1. LI-Fusion: 用图像语义特征增强点云特征；
2. 使用CE(consistency enforcing)损失函数提高定位精度和置信度；
3. 通过EPNet框架将LI-Fusion和CE损失函数集成起来，获取SOTA结果；

EPNet包含两个部分：

+ 双流RPN：几何流和图像流分别产生点云特征和图像语义特征，然后用多个LI-Fusion模块对点云特征进行增强。基本流程如下图所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1teB6RxAQlt)
+ 其中，LI-Fusion的结构如下图所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1teFwtmaGNk)
+ box细化(box refining)网络：

实验结果：KITTI和SUN-RGBD上SOTA。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1teI6A8HdHF)
代码：https://github.com/happinesslz/EPNet



## EPNet++: Cascade Bi-directional Fusion for Multi-Modal 3D Object Detection

【方法分类】特征级融合

本文提出一个新的级联二向融合(Cascade Bi-direction Fusion)模块CB-Fusion和多模一致性损失(Multi-Modal Consistency loss)函数MC loss，从而构建EPNet++框架。其中：

+ CB-Fusion是用图像语义特征增强点云特征的模块，通过级联双向交互融合的方式实现；
+ MC loss可获得更可靠的置信度；

EPNet++整体架构还是双流RPN+3D框细化网络，具体如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tfFYxOG7Pt)

与EPNet的区别在于：LI-Fusion和CB-Fusion结合，进行LC特征融合；使用MC loss提高置信度；
CB-Fusion的示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tfIc3e4eBt)

MC loss示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tfJ5wnjK2D)



实验结果：在KITTI， JRDB，SUN-RGBD数据集上SOTA，在稀疏点云场景改善显著。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tfJO08KNHc)
代码：https://github.com/happinesslz/EPNetV2



## Voxel Field Fusion for 3D Object Detection

【方法分类】特征级融合

本文提出一种简单高效的多模3D检测框架VF Fusion(Voxel Field Fusion), 该方法通过将图像特征转换为voxel空间的射线来保障特征映射一致性。
主要贡献有：

+ 可学习采样器(learnable sampler)；
+ ray-wise fusion: 使用增补上下文进行特征融合；
+ 混合增强器(mixed augmentor): 进行特征对齐；


本文提出一种新的多模融合框架VFF(voxel field fusion), 其架构如下图所示：

point-to-ray的特征映射示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tpXhWNbOPA)



整体架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tphmOdeX3J)



实验结果：在KITTI和nuScene上SOTA.
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tpjE3YbnF2)
代码仓：https://github.com/dvlab-research/VFF



## CAT-Det: Contrastively Augmented Transformer for Multi-modal 3D Object Detection

【方法分类】特征级融合，结合Transformer

本文提出CAT-Det解决LC融合3D检测问题，其中CAT是Contrastively Augmented Transformer。
CAT-Det是一个双流结构：一路是PointFormer, 另一路是ImageFormer。 
然后通过CMT(Cross-Modal Transformer)将PT, IT结合，对LC特征进行交互，计算权重后融合，然后进行联合编码、解码。
此外，我们还提出OMDA(One-way Multi-modal Data Augmentation)数据增强技术。

总体框架如下图所示：![image-20230108094346377](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094346377.png)

其中PT架构如下图：![image-20230108094401151](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094401151.png)D

IT架构如下图：![image-20230108094414800](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094414800.png)

CMT架构如下图：![image-20230108094428278](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094428278.png)

OMDA核心思想有两点：

1. 图像增强比点云增强复杂得多、困难得多，因此只对点云做数据增强；
2. 采用SECOND中GT-Paste思路，从其他帧的点云数据中增强当前帧的点云，通过学习的方式找到前后帧数据的对应关系；

具体增强方式分为点云级增强和目标级增强，示意图如下所示：![image-20230108094438687](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094438687.png)



实验结果：KITTI上SOTA。![image-20230108094316139](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094316139.png)



## BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation

本文提出的BEVFusion框架不同于之前的范式，在view转换过程中采用最优化BEV池化技术，延迟减少超过40倍。

本文提出的BEVFusion在BEV下进行LC特征融合，然后提出BEV池化加速技术缩短时间消耗。

整体架构如下图所示：![image-20230108094757013](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094757013.png)

1. 基于各种考虑，本文将camera和lidar都转换到BEV视图，然后再融合。LC的BEV图如下所示：![image-20230108094848037](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094848037.png)

   将camera图像转到BEV的示意图如下所示：![image-20230108094906470](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094906470.png)

2. 图像和点云都转到BEV后，可以很方便地将二者结合起来，比如级联等简单算子。
   转换到BEV之后，还是难免有空间不对齐情况。针对此问题，我们采用基于卷积的BEV编码器来对偏移进行补偿。

3. multi-task head: 采用多任务head将BEV特征图融合起来，本文以语义分割和目标检测任务举例：
   检测任务借鉴TransFusion和center-based检测思路；语义分割借鉴CVT和focal loss的思路。

实验结果：在nuScene上SOTA.![image-20230108094727454](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108094727454.png)
代码仓：https://github.com/mit-han-lab/bevfusion



## 3D Dual-Fusion: Dual-Domain Dual-Query Camera-LiDAR Fusion for 3D Object Detection

【方法分类】基于Transformer的特征级融合

本文提出一种LC融合的3D检测方法：3D Dual-Fusion。
该算法的核心思想是降低LC特征不对齐程度，通过transformer编码将camera特征和点云voxel特征进行融合，具体做法如下：

1. 基于双查询的deformable注意力机制(dual query-based deformable attention)来交互融合两个传感器的特征；分别用L融合C，和用C融合L；
2. 在双查询解码前对voxel查询进行局部自注意力编码(local self-attention encoding); 先用自注意力机制对局部voxel进行编码，然后用双查询机制同时细化图像特征和voxel特征；

3D Dual-Fusion的总体框架如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1triOR6rGpE)

双查询不可变注意力机制DDA(Dual-query Deformable Attention)如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1triJ5ZNLTF)



实验结果：使用TransFusion的检测头，在KITTI和nuScene部分数据集上SOTA。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1trbfDMsACW)
![image.png](https://s.readpaper.com/T/1trbuWN1Cj2)



## Cross Modal Transformer via Coordinates Encoding for 3D Object Dectection

【方法分类】：基于Transformer的特征级融合

本文提出一种端到端的多模3D检测框架：CMT(Cross Modal Transformer).
本方法不用对输入做view变换，直接对原始图像和点云处理，输出3D框。

CMT架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tsQDA31W1R)
基本流程如下所示：

1. L和C数据分别送到各自的backbone；
2. 使用坐标编码(coordinates encoding)对两个backbone的输出进行编码；
3. 使用位置引导查询(position-guided query)进行transformer decoder，然后预测目标类别和3D框；
4. 使用基于point的查询降噪方法加速训练收敛；



CMT基本组件如下：

+ 坐标编码模块CEM(Coordinates Encoding Module

  CEM对backbone的输出编码位置信息，通过CEM实现LC数据的3D空间对齐。
  CEM又分为针对图像的CE和针对lidar的CE。

+ 位置引导查询生成器(Position-guided Query Generator)

+ 基于point的查询降噪PQD(Point-based Query Denoising)

  基于DN-DETR扩展，示意图如下所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tsUoepevRY)



实验结果：在nuScene上SOTA(73%)。
73没有SOTA吧，现在的榜首都76以上了。。。。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tsBHqr8ozY)
代码仓：https://github.com/junjie18/CMT



## Unifying Voxel-based Representation with Transformer for 3D Object Detection



## PETR: Position Embedding Transformation for Multi-View 3D Object Detection

【方法分类】基于多camera的基于Transformer的特征级融合

本文提出一种嵌入位置信息的transformer来进行多视角3D检测框架PETR(position embedding transformation)，核心idea是将3D空间的位置信息编码到图像特征中。
**基于多视角图像的3D检测？**

本文基于DETR进行3D检测，PETR与DETR的差异点如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1twFz530Alc)
主要差异点在于：避免了复杂了2D-3D映射和特征采样。

transformer是广泛应用于长期依赖建模的注意力模块。transformer中，特征常常嵌入了位置信息。
最近，DETR将transformer应用于2D检测领域。DETR中，每个目标都表示为目标查询(object query), 然后通过transformer解码与2D图像特征交互。

DETR的主要问题是收敛速度慢，因此基于DETR的演进方法致力于提高DETR的收敛速度，比如交叉注意力机制、encoder-only DETR、增加位置先验(position priors)(比如2D参考点、anchor point等)；

PETR整体架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1twNp9yHwFU)



实验结果：在nuScene数据集上SOTA。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1twD1Clh8CW)
【吐槽】这个对比验证不充分呀，有失大佬颜面。。。
代码仓：https://github.com/megvii-research/PETR



## TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers

【方法分类】基于Transformer的特征级融合

LC融合在自动驾驶领域越来越流行，但是当前的融合技术基本都存在一个问题：图像边缘、弱光照场景、传感器不对齐场景的鲁棒性不够好，在这些场景下lidar的点云和图像的像素点匹配经常容易出错。
针对图像边缘匹配问题，我们提出一种称为TransFusion的软关联机制。
TransFusion的结构如下：

+ 卷积backbone；
+ 基于transformer decoder的detection head;

TranfFusion的融合机制如下：

1. decoder的第一层用稀疏集合查询机制从激光点云中预测初始边界框；
2. decoder的第二层利用空间信息和上下文信息将查询对象和图像特征自适应融合起来；

transformer的注意力机制使得我们的模型可以自适应选择所需的图像信息，这使我们的模型高效并且鲁棒性良好。
针对点云中难以检测的目标，我们还设计了基于图像引导(image-guide)的初始化查询策略。



### TransFusion细节

先对点云和图像进行卷积backbone处理，然后用基于transformer的detection head对lidar检测初始目标框，再将LC特征进行融合。
TransFusion架构图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tjCuGjYi2K)

### 1. 基于transformer的2D目标检测

Transformer的概念在2017年首次提出：**Attention is All you Need(cite50000, 2017)**, 之后被广泛用于目标检测，比如：

+ Fast Convergence of DETR with Spatially Modulated Co-Attention(cite100, 2021)
+ Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
+ Efficient DETR: Improving End-to-End Object Detector with Dense Prior
+ Deformable DETR: Deformable Transformers for End-to-End Object Detection
+ End-to-End Object Detection with Transformers

### 2. 查询初始化(Query Initialization)

Transformer中的query方法最早是随机生成的，最近有更高效的query方法，如：Efficient DETR: Improving End-to-End Object Detector with Dense Prior(cite50, 2021). 本文借鉴该方法的思想，也提出一种query initialization方法。

### 3. Transformer decoder和FPN

### 4. LC融合

+ 图像特征获取：使用交叉注意力机制(cross attention)在transformer decoder层执行自适应特征融合SMCA，示意图如上所示；    

+ SMCA(spatially modulated cross attention)融合：multi-head attention机制广泛应用于特征软关联领域和特征匹配领域，如：

  + SuperGlue: Learning Feature Matching with Graph Neural Networks；
  + LoFTR: Detector-Free Local Feature Matching with Transformers

  本文也用了特征软关联机制，即SMCA，从而实现动态查询融合的位置和融合所需信息。具体实现方式是：使用2D循环高斯掩码对交叉注意力加权重，示意图如下所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tjPjBGdSYi)
  SMCA之后，使用FPN获取目标框预测信息。

###  5. Label Assignment和损失函数

通过匈牙利算法匹配预测结果和真值；损失函数是分类、回归、IOU的权重和，具体公式如下：
    ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tjRZkmeHQo)

### 6. 图像引导查询初始化(Image guided query initialization)

之前的query只用lidar特征，我们提出一种结合图像特征和点云特征的query策略，即image-guided query initialization, 该查询机制示意图如下：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tjU9MuyIN7)



本文提出的TransFusion模型在大数据集上性能SOTA，并且在nuScenes的3d跟踪榜单上排名第一。
![image-20230108100129500](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108100129500.png)
本文首次提出将transformer应用于LC融合3D目标检测。

代码仓：[TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://github.com/XuyangBai/TransFusion/)





## DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection

【方法分类】基于Transformer的特征级融合

将图像特征与点云特征融合的难点在于，如何找到相对应的特征？深度学习运算时会频繁将各特征组合、激活等，这导致很难找到L和C中相对应的特征。

我们提出两个新技术来找到LC相对应的特征：

1. 反增广(InverseAug): 与旋转等几何增广操作相反；
2. 可学习对齐(LearnableAlign): 在特征融合过程中，使用交叉注意技术(cross attention)来动态寻找点云特征和图像特征的关联关系；

基于以上的反增广技术和对齐可学习技术，我们开发了名为DeepFusion的通用多模3D检测模型。

 DeepFusion架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkH1Ug0eeH)

我们将L和C的特征进行融合，而不是用C的特征修饰点云数据。
DeepFusion核心在于特征对齐，文章先分析了特征对齐的重要性，然后提出两种技术(InverseAug 和LearnableAlign)来提高特征对齐质量。

+ InverseAug:  该技术的目的是将点云映射到图像中，但是直接根据LC的参数进行映射结果不准确，InverseAug的做法是先找到lidar点云中所有关键点(如voxel的顶点)，然后根据LC参数将L的关键点映射到图像中。InverseAug示意图如下所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkM6Lfdj8a)
  InverseAug的效果如下图所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkMhkWnfzU)
+ LearnableAlign: 点云和像素之间的对应关系是一对多的，处理一对多映射的朴素方法是求平均，但是不是每个像素都同等重要。本文采用LearnableAlign技术解决一对多问题，采用交叉注意力机制(cross attention mechanism)来动态获取对应关系，示意图如下所示：
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkO4Gb7VtQ)



测试结果：比之前的方法都好, 精度高，鲁棒性好：

+ 提高了PointPillars, CenterPoint, 3D-MAN在行人检测的baseline, ；
+ 在Waymo开源数据集上SOTA;
  ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkPg2E45E8)
  代码仓：https://github.com/tensorflow/lingvo/ tree/master/lingvo



## FUTR3D: A Unified Sensor Fusion Framework for 3D Detection

本文设计了第一个端到端的用于3D目标检测的多传感器融合框架FUTR3D(Fusion Transformer for 3D Detection)，该框架几乎可用于任何多传感器配置场景。
FUTR3D主要技术点有两个：

1. 基于查询的多模不可知特征采样器(MAFS, Modality Agnostic Feature Sampler);
2. 采用set-to-set损失的transformer解码器用于3D检测；

FUTR3D的基本处理流程是：

1. 先将各传感器的原始信息独立编码；
2. 采用基于查询的未知模态特征采样器(MAFS, modality agnostic feature sampler)提取不同模态的特征；
3. 用解码器从3D查询集合里获取目标信息；



FUTR3D主要分为4个部分：

+ 对每个传感器的原始数据进行特征编码：对lidar采用voxelNet或PointPillar作为backbone，FPN为head提取点云特征；对图像采用ResNet+FPN提取图像特征；
+ 基于查询的异构模态特征采样器MAFS(Modality Agnostic Feature Sampler): 对多模信息进行特征聚合和采样；MAFS首先使用MLP对多模特征进行编码，然后添加位置编码，最后更新query;
+ 使用共享transformer decoder head处理融合特征，进行3D框细化；
+ 基于set-to-set的损失函数计算预测框和真值的损失值；

FUTR3D的架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkUYtr90oy)



实验结果：

以camera，低分辨率lidar，高分辨率lidar, radar各种组合，在NuScenes数据集上表现优于其他方法，同时配置灵活，成本低。
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tkdfU5Ouu1)





## CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection

【方法分类】后融合，基于两路proposal的融合

本文提出一种新的LC目标候选融合网络CLOCs(Camera-Lidar Object Candidates), 该框架简单，核心思想是 在NMS前进行候选目标融合，同时利用几何信息和语义信息产生尽量准确的预测结果。

CLOCs受两方面启发：

1. camera和lidar分别检测出2D和3D目标后，利用L和C的标定参数，可以将3D框准确地映射到图像上。示意图如下所示：
   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1uMLeiSGDMv)
2. 检测候选融合的理由：
   + 前融合简单但是存在对齐问题，效果一般；
   + 特征级融合太复杂；
   + 后融合不需要数据层面的对齐，架构简单；



CLOCs实现细节

1. 几何信息和语义一致性：融合候选目标需要做多模检测结果的关联，本文解决此问题的方法是应用语义一致性的同时计算几何关联的分数。具体方式如下：

   + 几何一致性：将3D框映射到2D平面，计算其IOU; 如果都是positive样本，则重叠区很大；如果都是误检，则重叠的概率很小；

   **【问题】：如果有一方漏检怎么办？**

   + 语义一致性：只融合同一分类的目标；

2. 网络架构：CLOCs架构示意图如下所示：
   ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1uMTtIRwrHk)
   组要组件有如下几个：

   + 将所有2D框和3D框转换为候选检测框的数据形式，方便输入融合网络框架；
   + 在NMS之前将所有候选框输入融合框架，假设k个2D候选，n个3D候选，则构建一个k*n*4的tensor T，每个tensor包含iou, 2D置信度，3D置信度，代价距离；具体tensor形式如下：![image.png](https://s.readpaper.com/T/1uMcgv8yr4r)
   + 融合模块只处理2D，3D均检测到的候选目标，只有一方检测到的目标，另一方的置信度和IOU置为-1；
   + 使用4层卷积，每层卷积之后ReLU;
   + 使用交叉熵损失函数，由focal loss进行损失修正；



实验结果：在KITTI上SOTA，尤其是远处目标表现优秀。 
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1uMpvisISxt)
![image.png](https://s.readpaper.com/T/1uMqIqAKFJh)



# 关于LC融合算法的总结

3D检测很重要，发展很快。本文review自动驾驶领域的3D检测技术发展历程。全文包含：

1. 3D检测的背景和挑战；
2. 从传感器模态和输入角度对3D检测技术做全面调查，并且分析每种模态技术的潜力和挑战，输入模态包括：
   + 基于lidar的技术；
   + 基于camera的技术；
   + 基于多模态的技术；

## 介绍

为了全面感知驾驶环境，感知系统的很多任务基于视觉技术，比如目标检测、跟踪、车道检测、语义分割、实例分割，而3D目标检测是感知系统最重要的任务之一。
3D目标检测的目的包括：

+ 预测目标的位置；
+ 预测目标的尺寸；
+ 关键目标的分类，如行人、机动车、自行车等；

与2D目标检测相比，3D检测更注重目标的位置和真实世界中的3D坐标。通过3D检测结果的几何信息，可以获得自动驾驶车辆与感知目标的距离等关键信息。

当前3D检测技术发展迅速，但是缺少各方法间综合的分析及系统的对比。
本文全面review了3D检测技术，并提出了深入分析及系统对比各种技术的方法。

1. 当前已有的3D检测综述有：

   + A Survey on 3D Object Detection Methods for Autonomous Driving Applications(Eduardo Arnold, cite300, 2019)
   + A survey of 3D object detection(Wei Liang, cite7, 2019)
   + 3D Object Detection for Autonomous Driving: A Survey(Xirong Li, cite0, 2022)

   相比已有的综述，我们的综述对3D检测技术的最新进展介绍得更全面，包括：基于range image的3D检测；自监督/半监督/弱监督的3D检测方法；端到端的3D检测方法；



2. 当前已有的基于lidar点云的3D检测算法综述有：
   + Deep Learning for 3D Point Clouds: A Survey([Bennamoun Mohammed](https://www.aminer.cn/profile/bennamoun-mohammed/53f43b44dabfaee2a1d16007)，cite870, 2021)
   + Point-cloud based 3D object detection and classification methods for self-driving applications: A survey and taxonomy(Daurte Fernandes,  cite0, 2021)
   + A comprehensive survey of LIDAR-based 3D object detection methods with deep learning for autonomous driving(Geofgios Zamanakos, cite20, 2021)
3. 当前已有的基于单目图像的3D检测算法综述：
   + A Survey on Monocular 3D Object Detection Algorithms Based on Deep Learning(Junhui Wu, cite0, 2020)
   + 3D Object Detection from Images for Autonomous Driving: A Survey(Xinzhu Ma, cite0, 2021)
4. 当前基于多模输入的3D检测算法综述：
   +  Multi-Modal 3D Object Detection in Autonomous Driving: a Survey(Yingjie Wang, cite0, 2021)

相比以上基于特定输入的综述，本文的综述范围包括各类传感器输入。

**【注】这个文章提到的综述还很不全。**



### 3D检测与2D检测对比

2D检测的目的是在图像中产生2D目标框，3D目标检测方法借鉴了很多2D检测的范式，比如：

+ proposal的产生和细化；
+ anchor理念；
+ 非极大抑制；

然而，3D检测不是2D检测方法在3D空间上的做简单的朴素适应，原因如下：

1. 3D检测方法必须处理异构数据：基于点云的检测方法需要新的运算符和网络，而且无论是基于点云的还是基于图像的3D检测方法都需要特殊的融合机制；
2. 3D检测方法通常需要利用不同的投影视图来成成目标框：2D检测方法通常从透视图中检测目标，而3D检测方法常常需要从不同的视图中检测目标，比如BEV, 点视图(point view)， 圆柱形视图(cylindrical view)等；
3. 3D目标检测对目标在3D空间中的定位精度有很高的要求：通常定位误差要小于分米级；



### 与室内场景的3D目标检测对比

关于室内场景3D目标检测的一些方法：

+ Frustum PointNets for 3D Object Detection from RGB-D Data([Leonidas J. Guibas](https://arxiv.org/search/cs?searchtype=author&query=Guibas%2C+L+J), cite1900, 2018)
+ Deep Hough Voting for 3D Object Detection in Point Clouds([Leonidas J. Guibas](https://arxiv.org/search/cs?searchtype=author&query=Guibas%2C+L+J),  cite300, 2019)
+ ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes([Leonidas J. Guibas](https://arxiv.org/search/cs?searchtype=author&query=Guibas%2C+L+J), cite160, 2020)
+ Group-Free 3D Object Detection via Transformers(Xin Tong, cite60, 2021)

室内3D检测数据集：标记了房屋的3D结构及窗户、桌椅、床等信息；数据集如下：

+  ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes
+  SUN RGB-D: A RGB-D scene understanding benchmark suite

与室内3D检测场景相比，驾驶场景的3D检测任务有一些特殊的困难，比如：

+ 检测范围大：驾驶场景一般要求检测范围是$150*150*6m$, 比如Waymo数据集；室内场景的检测范围一般是$10*10*3m$，比如ScanNet数据集；检测范围的不同导致一些适用于室内场景的高复杂度算法不适用于驾驶场景；
+ lidar和RGB-D传感器在驾驶场景产生的点云分布与室内场景不同：室内场景每个物体表面能收到足够的点云数量，但是驾驶场景中，远处的目标只能收到很少的点云。这导致驾驶场景需要处理远处点云稀疏的目标，要应对不用密度的点云；
+ 驾驶场景对检测时延要求更高：驾驶场景的检测速度必须满足实时性，否则会导致事故；

### 数据集

关于驾驶场景的数据集有很多，这些数据集都有多模传感器数据，并且做了3D标注。简要信息如下表所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1r6VArcTUVl)

其中KITTI数据集是最早的3D检测数据集，该数据集采集时，在车上安装了camera和lidar，在道路上采集数据之后，在数据中标注3D目标。
之后的数据集主要在以下四个方面改善：

1. 增加数据规模：相比于KITTI，以下数据集至少增加了10倍点云数据、图像数据和标注数据：
   + Scalability in Perception for Autonomous Driving: Waymo Open Dataset
   + nuScenes: A multimodal dataset for autonomous driving
   + One Million Scenes for Autonomous Driving: ONCE Dataset
2. 改善数据的分布：KITTI只采集了晴天和白天的数据，以下数据集还采集了晚上和雨天的数据：
   + KAIST Multi-Spectral Day/Night Data Set for Autonomous and Assisted Driving
   + Argoverse: 3D Tracking and Forecasting With Rich Maps
   + A*3D Dataset: Towards Autonomous Driving in Challenging Environments
   + nuScenes: A multimodal dataset for autonomous driving
   + Scalability in Perception for Autonomous Driving: Waymo Open Dataset
   + PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving
   + One Million Scenes for Autonomous Driving: ONCE Dataset
   + Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting
3. 标注了更多目标种类：包括动物、障碍物、交通锥，甚至标注了某些类别的子类，如大人和小孩等：
   + KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D
   + PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving
   + A2D2: Audi Autonomous Driving Dataset
   + Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting
   + nuScenes: A multimodal dataset for autonomous driving

4. 提供更多模态的数据，如高精度地图、radar数据、long-range lidar数据、热成像数据等：
   +  Lyft level 5 av dataset 2019. https:// level5.lyft.com/dataset/
   +  Argoverse: 3D Tracking and Forecasting With Rich Maps
   +  Scalability in Perception for Autonomous Driving: Waymo Open Dataset
   +  Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting
   +  nuScenes: A multimodal dataset for autonomous driving
   +  All-in-one drive: A large-scale comprehensive perception dataset with high-density long-range point clouds
   +  Cirrus: A Long-range Bi-pattern LiDAR Dataset
   +  KAIST Multi-Spectral Day/Night Data Set for Autonomous and Assisted Driving

**未来3D检测数据集的发展方向**
作者认为，未来3D检测数据集会演进为端到端的数据集，包括感知、预测、规划、定位等一系列信息。

## 评价指标

关于3D检测的评价方法有很多，这些方法大体分为两类：

1. 将2D检测指标AP(Average Percision)扩展到3D：![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rApJR1GQk5)
   该方法与2D指标的区别在于计算精准率和召回率时，真值与预测值是否匹配的规则不同，KITTI数据集提出两个广泛应用的指标：

   + $AP_{3D}$: 3D框IOU大于阈值则认为匹配上；
   + $AP_{BEV}$: 如果BEV视图的IOU大于阈值则认为匹配上；

   此外还有其他的匹配方法，比如基于中心点的匹配和匈牙利匹配等。

2. 基于实用主义角度的指标：核心思想是指标测量方法要与下游任务相关，基于此思想的指标有：

   + PKL：基于当前测量值与真值的规划导致的KL分歧，参考文献为Learning to Evaluate Perception Models Using Planner-Centric Metrics；
   + SDE: 利用自车与3D目标框的距离参数，度量距离误差，参考文献为Revisiting 3D Object Detection From an Egocentric Perspective；

### 不同评价指标的比较

+ 基于AP的度量方法继承了2D AP指标的优点，但是忽略了检测结果在安全方面的影响(远处检测不准与近处检测不准对于该方法没什么区别，但是实际自动驾驶场景中，近处检测不准的影响更大)；
+ PKL需要预训练运动规划器来计算检测指标，但是运动规划器本身也有误差；
+ SDE需要重建目标的3D框；

## 基于lidar的3D检测方法

本节介绍基于lidar数据的3D检测方法，即基于点云或range图像的3D检测方法。具体安排如下：

1. 3.1节从数据形式的角度介绍和分析基于lidar数据的3D检测模型，包括：
   + **基于点云的方法**；
   + **基于grid的方法**；
   + **基于point-voxel的方法**；
   + **基于range的方法**；
2. 3.2节从学习方式的角度介绍各种3D检测模型，包括：
   + **基于anchor的方法**；
   + **anchor-free的框架**；
   + **基于lidar的3D检测中的一些辅助任务**；

基于lidar的3D检测方法的里程碑如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rAz2e0xaHg)

### 3D检测的数据表现形式

图像的像素分布均匀，点云分布稀疏且不均匀，特征提取模型需要特殊设计。range图像稠密，但是呈现的是3D信息，而不是RGB信息，因此直接在range图像上使用卷积神经网络可能不会得到好的效果。
另一方面，自动驾驶场景的检测通常要求实时性，因此如何在满足实时性的同时满足高精度的检测要求仍是自动驾驶领域未攻克的难题。

### 基于点云的3D检测方法

lidar的工作原理是：不断发射激光，然后收集反射点信息，进而构建完整的环境信息。lidar通常是16-128channel的，即有16束激光到128束激光。每次扫描后会转动一定角度，进行下一次扫描。
市面上热门的64channel lidar大概价格是80000美金，所以用于自动驾驶的lidar一般小于64channel。



基于点云的3D检测技术一般都是将已有的深度学习技术应用到点云上，代表方法有：

+ PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation([Leonidas J. Guibas](https://www.aminer.cn/profile/leonidas-j-guibas/53f5669cdabfae631af8045b), cite10000, 2017)
+ PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space([Leonidas J. Guibas](https://www.aminer.cn/profile/leonidas-j-guibas/53f5669cdabfae631af8045b), cite6500, 2017)
+ Dynamic Graph CNN for Learning on Point Clouds([Michael M. Bronstein](https://www.aminer.cn/profile/michael-m-bronstein/53f7ce7ddabfae92b40e594c), cite13, 2019)
+ Interpolated Convolutional Networks for 3D Point Cloud Understanding([Hongsheng Li](https://www.aminer.cn/profile/hongsheng-li/53f46582dabfaee2a1dab2ad), cite200, 2019)

基于点云的3D检测框架如下：

1. 点云数据先经过基于点的backbone网络，点云算子对点逐步采样并学习特征；
2. 基于下采样点和特征进行3D框预测；

框架示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rCD8Ao2aLA)

该框架有两个基本组件：

1. 点云采样：PointNet++中提出的FPS(Furthest Point Sampling)采样方法被其他基于点云的检测方法广泛使用，比如：
   + PointRCNN首次应用FPS将输入点云逐步下采样，然后从下采样点中生成3D proposal;
   + IOPD在FPS范式的基础上引入了分割指导滤波(segmentation guided filtering);
   + 3DSSD在FPS范式的基础上引入了特征空间采样(feature space sampling);
   + StarNet在此范式的基础上引入了随机采样；
   + PointGNN在此范式的基础上上引入了基于voxel的采样；
   + PointFormer在此范式的基础上引入了坐标细化(coordinate refinement);

2. 特征学习：很多文章在PointNet的基础上进行抽象设置(set abstraction)来学习特征, 通常先通过预定义半径的球查询(ball query)来搜集上下文点(context points), 然后使用多层感知机和最大池化将上下文点和特征进行聚合来获得新的特征。此外还有很多不同的点云算子，比如：
   + 图算子(graph operators):  如Point-GNN， PointRGCN, StarNet, SVGA-Net, 
   + 注意力算子(attentional operators): 如Attentional PointNet;
   + Transformer: 如Pointformer

涉及到上述方法的文章有：

+ PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud([
  ,Xiaogang Wang](https://www.aminer.cn/profile/xiaogang-wang/5a28a3b39ed5db70fba3ff6c), [Hongsheng Li](https://www.aminer.cn/profile/hongsheng-li/53f46582dabfaee2a1dab2ad), cite1300, 2019)
+ IPOD: Intensive Point-based Object Detector for Point Cloud([Jiaya Jia](https://www.aminer.cn/profile/jiaya-jia/561b16d045cedb3397ef06c0), cite100, 2018)
+ STD: Sparse-to-Dense 3D Object Detector for Point Cloud([Jiaya Jia](https://www.aminer.cn/profile/jiaya-jia/561b16d045cedb3397ef06c0), cite500, 2019)
+ 3DSSD: Point-based 3D Single Stage Object Detector([Jiaya Jia](https://www.aminer.cn/profile/jiaya-jia/561b16d045cedb3397ef06c0), cite400, 2020)
+ Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud(Raj Rajkumar, cite400, 2020)
+ StarNet: Targeted Computation for Object Detection in Point Clouds([
  Shlens Jonathon](https://www.aminer.cn/profile/shlens-jonathon/53f4376fdabfaedd74da9d91), [Zhifeng Chen](https://www.aminer.cn/profile/chen-zhifeng/5d415bc57390bff0db709334), cite70, 2019)
+ 3D Object Detection with Pointformer([Li Li(Erran)](https://www.aminer.cn/profile/li-erran-li/53f43999dabfaefedbae66d1), [Gao Huang](https://www.aminer.cn/profile/gao-huang/540835d9dabfae44f0870362), cite120, 2021)
+ Joint 3d instance segmentation and object detection for autonomous driving([Ruigang Yang](https://www.aminer.cn/profile/ruigang-yang/562ffb0f45cedb33998a4b22), cite60, 2020)
+ 3d-centernet: 3d object detection network for point clouds with center estimation priority(Qi Wang, cite10, 2021)

+ PointRGCN: Graph Convolution Networks for 3D Vehicles Detection Refinement(Ghanem Bernard, cite3, 2019)
+ SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds(Zeng Bing, cite30, 2022)
+ Relation Graph Network for 3D Object Detection in Point Clouds(Mian Ajmal, cite30, 2021)
+ Attentional PointNet for 3D-Object Detection in Point Clouds(Christian Wolf, cite30, 2019)

对上述典型方法的分类如下表所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rMSfyaXBK4)

### 基于点云的3D检测方法总结

基于点云的3D检测方法效果主要受三个方面制约：

1. 上下文点(context point)的数量：增加上下文点的数量会获得更多的信息，但是会占用更多存储空间；

2. 特征学习过程中设置的上下文查询半径：查询半径过小会倒是上下文信息不足，半径过大会导致细粒度信息缺失；

   上下文点的数量和查询半径是需要仔细权衡的两个关键因素，这两个因素会显著影响检测模型的效率和精度。

3. 点云采样方法是基于点云的3D检测算法的瓶颈，不同的采样方法特点不同：

   + 随机均匀采样可以高效并行进行，但是会导致近处的点云过采样，远处的点云欠采样；
   + FPS(Furthest point sampling)在近远点的采样平衡性会更好，但是无法并发进行，不适合实时场景；

### 基于grid的3D检测方法

基于grid的方法基本框架如下：

1. 先将点云数据转换为离散grid；
2. 使用传统的2D卷积神经网络或3D稀疏神经网络从gird中提取特征；
3. 从BEV grid中检测3D目标；

基于grid的检测流程示意图如下所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rMTKmZqicC)



基于grid的3D检测框架有两个基本组件：

1. 基于grid的数据形式：基于grid的数据表示形式有3种：

   + voxels: voxel是3D立方体，每个立方体内包含点云。通过voxelization可以很容易将点云转换为voxels。因为点云分布是稀疏的，所以很多voxel里面是空的，没有任何点云。实际应用种，只会存储非空的voxel，然后从非空的voxel中提取特征。

     + 首次提出voxel概念的文章：VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection([Oncel Tuzel](https://www.aminer.cn/profile/oncel-tuzel/53f47b73dabfaee4dc89e681),  cite2000, 2017)
     + 类似voxel表示形式的文章：
       + Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection(Yu Gang, cite0, 2019)
       + AFDet: Anchor Free One Stage 3D Object Detection(Li Huang, cite60, 2020)
       + Center-based 3D Object Detection and Tracking([Philipp Krähenbühl](https://www.aminer.cn/profile/philipp-kr-henb-hl/53f4589ddabfaedd74e36b9a), cite350, 2021)
       + Object DGCNN: 3D Object Detection using Dynamic Graphs(Justin Solomon, cite20, 2021)
       + CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud(Chi-Wing Fu, cite100, 2021)
       + Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection(Yanyong Zhang, cite200, 2021)
       + From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network([Hongsheng Li](https://www.aminer.cn/profile/hongsheng-li/53f46582dabfaee2a1dab2ad), cite400, 2021)
     + 多视角的voxel方法：从range视图、圆柱视图、球状视图、感知视图、BEV视图等多种视图动态voxelization和融合的机制来生成voxel，代表性方法有：
       + End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds([Anguelov Dragomir](https://www.aminer.cn/profile/anguelov-dragomir/53f43947dabfaedce554a66b)， cite200, 2019)
       + Every View Counts: Cross-View Consistency in 3D Object Detection with Hybrid-Cylindrical-Spherical Voxelization(Qi Chen, cite40, 2020)
       + VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention(Kui jia, cite10, 2022)

     + 多尺度的voxel方法：voxel的尺寸不同，代表文章有：
       + HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection(Ye Maosheng, cite100, 2020)
       + Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds([Lin Dahua](https://www.aminer.cn/profile/lin-dahua/53f42cf2dabfaedce54bedd8), cite0, 2020)
     + pillars: pillar可以看作是一种特殊的voxel, 它的垂直方向是无限大。先通过PointNet将点云聚合成pillar特征，然后分散后构成2D BEV图像，之后对2D图像进行特征提取。
       + 首次提出pillar概念的文章：PointPillars: Fast Encoders for Object Detection from Point Clouds(Alex H Lang, cite1600, 2019)
       + 基于PointPillars的演进方法有：
         + Pillar-Based Object Detection for Autonomous Driving([Tom Funkhouser](https://www.aminer.cn/profile/tom-funkhouser/53f438fddabfaeecd6977b07), cite100, 2020)
         + Embracing Single Stride 3D Object Detector with Sparse Transformer(Zhaoxiang Zhang, cite30, 2022)
     + BEV特征地图(BEV feature map):  BEV特征图是种稠密的2D表示，每个像素对应一个特定的区域，并对该区域内的点信息进行编码。可以通过将voxel或pillar的3D特征映射到BEV或者通过统计像素区域内的点的信息来产生BEV特征图。基于BEV的3D检测方法有：
       + PIXOR: Real-time 3D Object Detection from Point Clouds([R Urtasun](https://scholar.google.com/citations?user=jyxO2akAAAAJ&hl=zh-CN&oi=sra), cite700, 2018)
       + HDNET: Exploiting HD Maps for 3D Object Detection([R Urtasun](https://scholar.google.com/citations?user=jyxO2akAAAAJ&hl=zh-CN&oi=sra), cite200, 2018)
       + RAD: Realtime and Accurate 3D Object Detection on Embedded Systems(Robert Laganiere, cite2, 2021)
       + Multi-View 3D Object Detection Network for Autonomous Driving(Xiaozhi Chen, cite1800, 2016)
       + BirdNet: A 3D Object Detection Framework from LiDAR Information(Barrera Alejandro, cite180, 2020)
       + YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud(Waleed Ali, cite80, 2018)
       + Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds(Horst-Michael Gross, cite200, 2018)
       + Vehicle Detection from 3D Lidar Using Fully Convolutional Network(Bo Li, cite500, 2016)

     以上提到的基于grid的数据形式的3D检测方法分类如下：
     ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rTAq63nHFo)

​            

### 基于grid的神经网络

主要有两种基于grid的神经网络：

1. 用于BEV特征图和pillar的2D卷积神经网络：2D神经网络一般都能很好适配2D目标检测任务，比如：
   + 将[ResNet(Kaiming He, cite145000, 2016)](https://ieeexplore.ieee.org/document/7780459/)用于PIXOR;
    + 将[RPN(Region Proposal Network)(Kaiming He, cite50000, 2017)](https://ieeexplore.ieee.org/document/7485869)和[FPN(Feature Pyramid Network)(Kaiming He, cite10000, 2016)](https://www.connectedpapers.com/main/b9b4e05faa194e5022edd9eb9dd07e3d675c2b36/Feature-Pyramid-Networks-for-Object-Detection/graph)用于如下论文：
      + BirdNet;
      + BirdNet+;
      + PointPillars;
      + Voxel-FPN;
      + Complex-YOLO;
      + PSANet;
      + Anchor-free 3D Single Stage Detector with Mask-Guided Attention for Point Cloud;
      + TANet: Robust 3D Object Detection from Point Clouds with Triple Attention([Xiang Bai](https://www.aminer.cn/profile/xiang-bai/53f45a3ddabfaee02ad67536), cite150, 2020)
      + SARPNET: Shape attention regional proposal network for liDAR-based 3D object detection(Zhaoxiang Zhang, cite50, 2020)

2. 用于voxel的3D稀疏神经网络：3D稀疏卷积神经网络基于两种特定的3D卷积算子：稀疏卷积和[submanifold](https://www.connectedpapers.com/main/1d6a5d0299ed8458191e4e0407d4d513e6a7dd7e/3D-Semantic-Segmentation-with-Submanifold-Sparse-Convolutional-Networks/graph)卷积；这两种卷积算子可以只对非空voxel进行高效3D卷积。相比于对所有voxel都进行卷积的网络，使用稀疏卷积算子可以极大提高运算效率，满足实时要求。基于这两种3D卷积算子的神经网络框架已成为基于voxel的3D检测领域中最广泛应用的backbone网络。
   + 首次使用稀疏卷积和submanifold卷积的文章：SECOND: Sparsely Embedded Convolutional Detection(Bo Li, cite1100, 2018)
   + 基于SECOND的演进方法：
     + Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection(Gang Yu, cite200, 2019)
     + Center-based 3D Object Detection and Tracking([Philipp Krähenbühl](https://www.aminer.cn/profile/philipp-kr-henb-hl/53f4589ddabfaedd74e36b9a), cite350, 2021)
     + Object DGCNN: 3D Object Detection using Dynamic Graphs(Justin Solomon, cite20, 2021)
     + AFDet: Anchor Free One Stage 3D Object Detection(Li Huang, cite70, 2020)
     + Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots([Yuille Alan](https://www.aminer.cn/profile/yuille-alan/53f43030dabfaeb2ac00d2ac), cite100, 2020)
     + SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds([Lin Dahua](), cite50, 2020)
     + Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection(Yanyong Zhang, cite200, 2021)
     + CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud(Chi-Wing Fu, cite100, 2020)        
   + 其他一些针对3D检测的方法：
     + Focal Sparse Convolutional Networks for 3D Object Detection([Jiaya Jia](), cite20, 2022)
     + From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network([Hongsheng Li](), cite400, 2021)
     + Voxel Transformer for 3D Object Detection([Jiashi Feng](), [Xiaodan Liang](), cite80, 2021 )
     + Embracing Single Stride 3D Object Detector with Sparse Transformer(Zhaoxiang Zhang, cite30, 2022)

### 基于grid的lidar 3D检测算法总结

1. voxel携带更多深度信息，但是对应的计算和存储资源消耗更多；
2. 基于BEV特征图的方法速度快，但是检测精度比基于voxel的低；
3. 基于pillar的精度和效率介于BEV和voxel之间；

基于grid方法面临的同一个问题是：如何选取一个合适的grid size？

+ grid size较小时，携带的信息粒度更精细，但是存储和计算量较大；
+ grid size较大时，存储和计算效率高，但是检测精度较低；

### 结合point和voxel的方法

point和voxel结合的检测方法的基本框架如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rU2fsUE5fE)
此类方法可分为两类：

+ one-stage方法：该方法通过point-voxel和voxel-point转换的方法在point和voxel之间建立关联；点云保留了细粒度的深度信息，voxel计算效率高，在特征提取阶段将二者结合。基于该思想的文章有：
  + Point-Voxel CNN for Efficient 3D Deep Learning([Han Song](https://www.aminer.cn/profile/song-han/542a5aeadabfae646d5557e9), cite300, 2019)
  + Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution(Han Song, cite200, 2020)
  + Structure Aware Single-Stage 3D Object Detection From Point Cloud([Lei Zhang](https://www.aminer.cn/profile/lei-zhang/53f4986fdabfaee0d9c74c66), cite300, 2020)
  + From Voxel to Point: IoU-guided 3D Object Detection for Point Cloud with Voxel-to-Point Decoder(Ling Shao, cite10, 2021)
  + From Multi-View to Hollow-3D: Hallucinated Hollow-3D R-CNN for 3D Object Detection([Yanyong Zhang](), cite10, 2021)
  + PVGNet: A Bottom-Up One-Stage 3D Object Detector with Integrated Multi-Level Features(Yang Wang, cite10, 2021)
  + HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection(Bumsub Ham, cite20, 2021)
  + M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers(
    [Larry Davis](), [Dinesh Manocha](https://www.aminer.cn/profile/dinesh-manocha/53f4b7d7dabfaed4ae77b475), cite3, 2022)

+ two-stage方法：每个阶段使用的数据形式不同，一般来说，第一阶段使用基于voxel的框架产生一系列3D proposal；第二阶段从点云中采样关键点，然后使用特定的point算子从关键点中细化3Dproposal。代表文章有：
  + PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection([Hongsheng Li](), cite800, 2020)
  + Fast Point R-CNN([Jiaya Jia](), cite200, 2019)
  + PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection([Hongsheng Li](), cite50, 2021)
  + InfoFocus: 3D Object Detection for Autonomous Driving with Dynamic Information Modeling([Larry S. Davis](https://www.aminer.cn/profile/larry-s-davis/53f49c81dabfaee1c0badc66), cite20, 2020)
   + LiDAR R-CNN: An Efficient and Universal 3D Object Detector(Zhihao Li, cite50, 2021)
   + Pyramid R-CNN: Towards Better Performance and Adaptability for 3D Object Detection([Xiaodan Liang](), cite40, 2021)
   + Improving 3D Object Detection With Channel-Wise Transformer(XianSheng Hua, cite40, 2021)
   + Point Density-Aware Voxels for LiDAR 3D Object Detection(Steven Waslander, cite10, 2022)

以上提到的各种方法的分类情况如下：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rUcgNtHRT7)

**point-voxel结合方法的总结**：
此类方法的检测精度比纯voxel的高，但是计算成本和难度增加，point-voxel的关联耗时且不准确，同时点云特征和3D proposal的配合也难度较大。

### 基于range的3D检测

range图像是一种稠密的2D表示，每个像素携带了距离信息，而不是RBG信息。基于range图像的3D检测方法主要从两个方面来完成3D目标检测任务：

1. 设计专用于range图像的模型和算子；
2. 选择合适的view；

基于range图像的3D检测框架如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rWAhtai3oO)

#### 基于range的模型

因为range和普通camera的图像都是2D的，所以常将2D检测模型移植到range图像上。

+ 开创性的工作：LaserNet，使用DLA-Net(deep layer aggregation network)来获得多尺度的特征，并从range图像中检测3D目标， 论文题目：LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving(Gregory P. Meyer, cite200, 2019)
+ 演进工作：
  + RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation（Zhidong Liang, cite40, 2020）
  + RSN: Range Sparse Net for Efficient, Accurate LiDAR 3D Object Detection([Drago Anguelov](), cite50, 2021)
  + Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection([Drago Anguelov](), cite20, 2020)
  + RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union(Zhidong Liang, cite20, 2021)
  + RangeDet: In Defense of Range View for LiDAR-Based 3D Object Detection(Zhaoxiang Zhang, cite60, 2021)

#### 基于range的算子

由于range图像和RGB图像中每个像素携带的信息不同，所以使用的算子也不同。一些方法通过改善算子来提高特征提取的效率和精度，代表性方法如下：

+ Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection([Anguelov Dragomir](), cite30, 2020)
+ To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels([Anguelov Dragomir](), cite20, 2021)

#### 基于range view的方法

range图像是由点云进行球状映射得到的，基于range视角的检测会存在遮挡和比例尺变化的情况。为解决此问题，很多文章尝试从其他view来进行3D检测，代表性方法有：

+ It's All Around You: Range-Guided Cylindrical Network for 3D Object Detection(Dan Raviv, cite20, 2021)

以上提到的各方法的分类如下：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rWWZLYaV1c)

#### **关于range 3D检测方法的总结**

基于range的检测方法效率高，但是容易受遮挡和尺度变化影响。
因此，当前流行的基于range的范式是从range中进行特征提取，从BEV进行3D检测。

### 基于anchor的lidar 3D检测

1. 3D目标比较小；
2. 点云稀疏，难以检测和3D目标尺寸估计；

anchor是预定义尺寸的长方体，可被置于3D空间的任意位置。
假设ground truth为$[x^g, y^g, z^g, l^g, w^g, h^g, \theta^g], cls^g$;
anchor为$[x^a, y^a, z^a, l^a, w^a, h^a, \theta^a]$;
由anchor预测得到3D检测框$[x, y, z, l, w, h, \theta]$.
基于anchor的3D检测基本流程如下图所示：![image-20230108091242557](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230108091242557.png)

我们从两个方面来介绍基于anchor的解决上述问题的技术：

1. anchor的配置：基于anchor的3D检测方法基本都基于BEV,3Danchor放置于BEV特征图中每个grid中，并且每个类别的anchor都有固定的尺寸；

   **问题：每个类别固定尺寸的anchor是否合理？**

2. 损失函数：$L_{det} = L_{cls} + L_{reg} + L_{\theta}$, 其中：

   + $L_{cls}$是positive anchor和negative anchor的分类回归损失；
   + $L_{reg}$是3D尺寸和位置的损失函数；
   + $L_{\theta}$是heading的损失函数；

   voxelNet是第一个将3D IOU和损失函数结合起来的文章，之后还有基于focal loss的方法，基于SmoothL1的方法，基于corner loss的方法等。

**基于anchor检测方法的挑战**：

1. 对于小目标，使用anchor难度大；
2. grid大时，anchor一般也大， 如果目标小则iou小；
3. 实际应用中，需要的anchor数量太多；

#### anchor-free的3D目标检测

anchor-free的检测方法不需要设计复杂的anchor机制，可以灵活应用于BEV, point view, range view等多种视图。基于anchor-free的3D检测基本流程如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rZcrTCCh5U)

anchor-free和anchor-based检测方法的根本区别在于正样本和负样本的选择上。
anchor-based方法基于IOU来决定正负样本；
anchor-free方法使用各种各样基于grid的assignment策略，这些策略适用于grid cell, pillars, voxels.

1. 基于grid的anchor-free方法：
   PIXOR是使用grid cell来决定正负采样的先驱：grid cell在ground truth里就是正样本，否则就是负样本。
   基于PIXOR的演进方法有：

   + Pillar-based Object Detection for Autonomous Driving；
   + AFDet;
   + AFDetV2;
   + Object as Hotspots
   + Center-based 3D Object Detection and Tracking

2. 基于point的anchor-free方法：
   基于point的3D检测方法大多采用基于point的anchor-free策略，先将point分段，然后将3D框内部或附近的前景点作为正样本，然后从这些前景点中学习3D检测框。采用这种方法的代表性文章有：

   + PointRCNN;
   + 3DSSD;
   + IPOD;
   + 3D Object Detection with Pointformer;

3. 基于range的anchor-free方法：

   基于range的anchor-free方法通常做法是：将3D目标内部的range像素作为正样本，此外不同于其他方法采用基于全局坐标的3D框回归，基于range的3D框回归使用的是以目标为中心(object-centric)的坐标。代表性方法有：

   + LaserNet;
   + RangeDet;

4. 集合到集合的anchor-free方法(set-to-set assignment)：
   DETR的set-to-set assignment方法影响深远，通过匈牙利算法自动将预测结果和ground truth关联起来。基于set-to-set的代表性文章有：

   + An End-to-End Transformer Model for 3D Object Detection
   + Object DGCNN: 3D Object Detection using Dynamic Graphs
   + Point2Seq: Detecting 3D Objects as Sequences

**关于anchor-free方法的总结**：
anchor-free方法灵活简单，各种anchor-free方法中，最具潜力的方法是center-based方法：Center-based 3D Object Detection and Tracking，其在小目标领域表现良好，并且超过anchor-based基线。
anchor-free方法的难点在于，需要设计一个准确的过滤机制，将bad positive样本过滤掉。在这个问题上，anchor-based方法只需要计算iou，设置iou阈值即可。

### 通过辅助手段来改善3D检测的方法

很多种方法都在尝试通过辅助工作来加强空间特征，以及为精确的3D检测提供辅助支持。常用的辅助工作包括：

1. 语义分割：语义分割可以从以下3个方面对3D检测提供帮助：
   1.  前景分割：可以提供目标位置的潜在信息，应用该方法的文章有：
       + PointRCNN;
       + Joint 3D Instance Segmentation and Object Detection for Autonomous Driving
       + STD: Sparse-to-Dense 3D Object Detector for Point Cloud
       + Structure Aware Single-Stage 3D Object Detection From Point Cloud
   2.  采用语义分割可以加强空间信息：
   3.  可以将语义分割作为预处理手段，过滤背景样本，提高检测效率。采用此方法的文章有：
       + IPOD；
       + RSN；
2. IOU预测：IOU可以作为一个有效的监督手段来纠正目标置信度。采用IOU辅助手段的文章有：
   + CIA-SSD：为每个检测到的3D目标计算IOU置信度$S_{iou}$ ,然后使用此置信度纠正推理结果，最终的置信度计算公式为$S_{conf} = S_{cls} * (S_{iou})^{\beta}$;
   + SE-SSD;
   + RangeIoUDet;
   + AFDet;
   + AFDetV2;
3. 目标形状补全：由于lidar的物理特性，导致远处的目标感知到的点云稀疏，形状不完整。解决此问题的一个直观方法是从稀疏点云中补全目标形状，进而得到精确和鲁棒的检测结果。关于目标补全技术的文章有：
   + DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes([Funkhouser Thomas](https://www.aminer.cn/profile/funkhouser-thomas/53f438fddabfaeecd6977b07), cite30, 2020)
   + SSN;
   + SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation([Dragomir Anguelov](), cite30, 2021)
   + Behind the Curtain: Learning Occluded Shapes for 3D Object Detection(Qiangeng Xu, cite10, 2021)
4. 目标局部预测：获取目标的局部信息有利于3D检测，相关的文章有：
   + Object as Hotspots：
   + From Points to Parts

**关于3D检测辅助手段的总结**：
还有很多辅助手段来提高检测精度，比如场景流估计(scene flow estimation)等。

## 基于多模的3D检测方法

lidar和camera是3D目标检测领域中两个重要的可以互补的传感器：

+ lidar点云带有低分辨率的形状和深度信息；
+ camera图像带有高分辨率的形状和纹理信息；

截至2021年10月，Waymo数据集关于3D检测榜单中，大部分SOTA模型只使用lidar点云数据，这表明当前的lidar点云和camera图像融合技术还不够成熟。

从2022年开始，随着几个基于Transformer的LC融合方法的问世，LC融合的指标慢慢开始反超基于单Lidar的指标。

基于多模的3D检测方法可以分为3类:

1. lidar-camera融合,具体包括:
   + 前融合;
   + 中融合;
   + 后融合;
2. radar融合;
3. 高精度地图融合;

### 基于LC融合的3D检测方法

lidar和camera属于特性互补的两种传感器, camera提供颜色信息, 从而可以提取处丰富的语义特征;lidar提供丰富的3D结构信息. 
关于LC融合进行3D检测的文章有很多,因为基于lidar的检测效果远远优于基于camera的效果, 所以大部分SOTA方法都是基于lidar的3D检测方法, 然后在不同阶段将camera图像信息集成到lidar检测的pipeline中.
由于分别基于lidar和基于camera的3D检测方法就很复杂,所以LC融合的方法计算更复杂,时延更大. 因此如何高效融合二者信息是该课题的一大挑战.
基于LC融合的3D检测方法分类情况如下图所示:
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rdEADtO9YG)



### LC前融合方法

LC前融合是指：先将camera的信息集成到点云中，然后再将融合数据送入基于lidar的pipeline中。
因此前融合框架基本都是以时序的方式构建：

1. 先用2D检测或分割网络从图像中提取特征；
2. 将图像信息送到点云中；
3. 增强的点云输入到基于lidar的检测器中；

早期的前融合技术可以分为两类：

1. 区域及的融合；
2. 点云级的融合；

这两类融合架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rdVVkhFEno)

1. 区域级前融合方法

   区域级融合的目的是利用图像的信息来减少点云检测的候选区域。
   一般流程是：

   1. 在图像上进行2D检测，得到2D检测框；
   2. 将2D检测框转换为3D视锥(3D view frustums);
   3. 用3D frustums减少lidar检测的候选空间；
   4. 将选出来的lidar点云区送入lidar 检测器进行3D目标检测；

   代表性文章有：

   + Frustum PointNets for 3D Object Detection from RGB-D Data([Leonidas Guibas]() , cite1500, 2017)
   + Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection(Kui Jia, cite300, 2019)
   + RoarNet: A Robust 3D Object Detection based on RegiOn Approximation Refinement([Masayoshi Tomizuka](https://www.aminer.cn/profile/masayoshi-tomizuka/53f43b47dabfaeb22f4a0a83), cite40, 2019)
   + Frustum-PointPillars: A Multi-Stage Approach for 3D Object Detection using RGB Camera and LiDAR
   + A General Pipeline for 3D Detection of Vehicles

2. 基于点云的LC前融合方法

   基于点云融合的目的是利用图像特征增强点云数据.  然后通过增强的点云来获得更好的检测结果.
   基于点云前融合的开创性工作是: **PointPainting: Sequential Fusion for 3D Object Detection(Vora Sourabh, cite300, 2020)**, 该方法的核心思想是:先获得图像的像素级语义标签, 然后建立像素-点云映射关系, 将像素标签增强到点云上, 然后将增强的点云送到lidar检测网络中. 
   演进的工作有:

   + FusionPainting: Multimodal Fusion with Adaptive Attention for 3D Object Detection(cite30, 2021)
   + Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds()
   + Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation(Gregory P. Meyer, cite100, 2019)
   + 3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-view Spatial Feature Fusion for 3D Object Detection

**关于LC前融合的总结**:
前期的LC融合技术偏向于用图像信息增强point之后再送到lidar检测网络中, 相当于是对point的预处理, 后面的检测网络几乎不用改.
但是由于该方法是时序性的方法, 需要先对图像进行2D检测或语义分割. 所以不可避免引入延迟, 所以前融合的一大改进方向是如何提高效率.

另一方面，所有融合问题都可以归结为关联问题。对于LC前融合，关键点在于图像数据与点云数据的关联关系往往不准确，所以前融合的效果也有限。

### LC中融合方法

中期融合方法主要是在lidar检测网络的backbone中, 将图像和点云的特征融合起来, 比如proposal生成阶段, 或者RoI细化阶段.
该融合方法的架构如下图所示:
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rgSbTuCfAm)

#### 基于backbone的LC中融合

该融合方法需要先建立pixel-point的对应关系,然后用各种融合算子将lidar backbone的特征与camera backbone的特征进行融合. 
该方法的代表性文章有:

1. 在backbone网络中提出各种融合算子的文章:
   + Deep Parametric Continuous Convolutional Neural Networks([Raquel Urtasun](), cite350, 2018)
   + Deep Continuous Fusion for Multi-Sensor 3D Object Detection([Raquel Urtasun](), cite550, 2018)
   + Multi-Task Multi-Sensor Fusion for 3D Object Detection([Raquel Urtasun](), cite500, 2019)
   + MVX-Net: Multimodal VoxelNet for 3D Object Detection([Oncel Tuzel](), cite100, 2019)
   + DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection([Alan Yuille](https://www.aminer.cn/profile/alan-yuille/53f43030dabfaeb2ac00d2ac), cite30, 2022)
   + CAT-Det: Contrastively Augmented Transformer for Multi-modal 3D Object Detection(Di Huang, cite10, 2022)
2. backbone输出特征图之后进行融合的方法:
   + 3D-CVF: Generating joint camera and lidar features using cross-view spatial feature fusion for 3d object detection(cite150, 2020)
   + FUTR3D: A Unified Sensor Fusion Framework for 3D Detection(cite20, 2022)
   + BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation(Song Han, cite30, 2022)
   + AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection(Zehui Chen, cite20, 2022)
   + Voxel Field Fusion for 3D Object Detection([Jiaya Jia](), cite10, 2022)
   + TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers([Hongbo Fu](), cite40, 2022)
   + SEG-VoxelNet for 3D Vehicle Detection from RGB and LiDAR Data(cite15, 2015)
   + Boost 3-D Object Detection via Point Clouds Segmentation and Fused 3-D GIoU-L₁ Loss
   + PointAugmenting: Cross-Modal Augmentation for 3D Object Detection([Xiaokang Yang](), cite50, 2021)

3. 基于grid的检测框架的特征级融合方法:
   + PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation([Dragomir Anguelov](), cite400, 2017)
   + EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection()
   + PI-RCNN: An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module(cite60, 2019)
   + Multi-Stage Fusion for Multi-Class 3D Lidar Detection(cite5, 2021)
   + Cross-Modality 3D Object Detection([Xiaokang Yang](), cite10, 2020)

#### 基于proposal和RoI head的中融合

在proposal生成和RoI细化阶段的特征级融合, 这类融合算法的基本处理流程是:

1. 由lidar检测网络生成3D proposal;
2. 将3D proposal映射到多视角, 比如图像视角, BEV, 然后从图像和lidar的backbone中提取特征;
3. 在RoI head中融合图像特征和lidar特征, 进而预测每个3D目标的参数;

代表性的文章有:

+ Multi-View 3D Object Detection Network for Autonomous Driving(Xiaozhi Chen, cite1700, 2016)
+ Joint 3D Proposal Generation and Object Detection from View Aggregation(Jason Ku, cite900, 2017)
+ FUTR3D: A Unified Sensor Fusion Framework for 3D Detection()
+ TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers([HongboFu](), cite40, 2022)

**基于LC特征级融合算法的总结**:
特征级融合需要LC信息深度融合,但是LC两种传感器是异构的, 并且视角也不同, 所以融合机制及视角对齐方面还存在较多问题,这两个问题也是这个领域当前的研究热点.

随着技术越来越成熟，特征级关联的解决方法也由一对多的硬关联慢慢过渡到以Transformer为代表的软关联，融合效果也越来越好。

### LC后融合方法

后融合指的是对3D检测框和2D检测框融合. 后融合基本框架如下图所示:
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1rhdniLRNWj)

LC后融合的代表性文章有:

+ CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection(Hayder Radha, cite100, 2020)
+ Fast-clocs: Fast camera-lidar object candidates fusion for 3d object detection(Hayder Radha, cite2, 2022)

**LC后融合方法总结**:
计算简单, 精度一般.



LC融合3D检测算法的里程碑如下图所示：

![image-20230118145528314](http://image.huawei.com/tiny-lts/v1/images/08a10b2b455d9de214611b275278dee5_1486x496.png)





# 附录



## Transformer结构(Attention is All you Need)

### 摘要

当前主流的针对序列数据的模型都包含复杂的回归或包含编解码器的卷积神经网络，性能好的模型甚至喊通过注意力机制连接编解码器。
本文提出一种完全基于注意力机制的简单架构：Transformer。

实验结果：在BLUE上SOTA。

### 背景介绍

序列数据模型的SOTA方法基本都是基于循环神经网络(RNN, Recurrent neural networks)、长短期记忆网络(LSTM, long short-term memory)、GRNN(gated recurrent neural networks)的方法。



### Transformer模型细节

很多针对序列数据的模型都有编解码模块，将输入$(x_1, x_2, ... , x_n)$编码为$(z_1, z_2, ... , z_n)$, 然后再用解码模块将z解码为$(y_1, y_2, ... , y_m)$. 
Transformer也使用此结构，然后使用堆叠自注意力机制(stacked self-attention) 和point-wise，并且对encoder和decoder都使用全连接层。Transformer的整体架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1twe9rcQIIj)

#### encoder stack

encoder模块使用6个相同的层，每个层包含两个子层：

+ 第一个子层：多头自注意力机制(multi-head self-attention mechanism)；
+ 第二个子层：position-wise的全连接前馈网络；

我们对encoder模块的每个子层进行残差连接(residual connection), 然后进行归一化。为方便残差连接，每个子层及嵌入层的输出维度均为512.

#### decoder stack

decoder也是6个相同的层，每个层有3个子层，除了对应于encoder的多头自注意力子层和全连接前馈层外，还增加了一层：对encoder stack的输出进行多头注意力操作。
同encoder层一样，decoder的每个子层也有残差连接和归一化。
此外，decoder的自注意力层也做了一点修改，防止关注后续位置。

#### 注意力机制

注意力函数的作用是将query和key-value集合映射到输出，其中query, key, value, output都是向量。输出是每个value的权重和，每个value的权重是通过查询与相应key的兼容性函数求得的。
多头自注意力机制的结构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1twmXBJWOAa)
注意力模块的输入包含key, value, 和query, 使用SoftMax函数计算value的权重。
使用线性映射将输入的key，value，query进行映射，然后并行执行注意力函数，



## 基于Transformer的E2E检测网络(End-to-End Object Detection with Transformers)

### 摘要

本文提出一种目标检测新方法DETR(Detection Transformer)，该方法将目标检测问题转换为集合预测问题。
该方法不需要涉及NMS或anchor生成模块，由基于集合的全局损失函数和transformer的编解码模块组成。

经COCOs数据集测试，DETR和当前非常成熟的Faster RCNN表现相当。
代码仓：https://github.com/facebookresearch/detr

### 背景介绍

目标检测任务的目的是预测一系列边界框及类别标签的集合。
当前的检测器使用间接的方法达成这个目标，比如基于proposal、anchor、中心窗(window centers)等, 这些检测方法的效果受到上述设计的极大影响。

为简化上述pipeline, 我们提出一种直接进行集合预测的方法。
本文基于transformer提出一种编解码架构，其中transformer的自注意机制模拟了元素间的相互作用，使得此架构适合消除重复预测等约束。
此外，DETR不需要对anchor、NMS等先验信息人工编码，不需要任何自定义层。
DETR基本架构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1txHzvqmnWT)

DETR在COCO数据集上表现效果与Faster-RCNN相近，并且对大目标的检测精度很高，但是对小目标的检测效果较差。我们希望后续的研究能像FPN改进Faster-RCNN一样改进DETR。

### DETR模型细节

DETR主要包含两个关键创新点：

1. 强制预测结果与真值匹配的预测损失函数；
2. 预测目标集合及其相关联系的模型；

DETR具体结构如下图所示：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1txMbV9rSwz)
主要包含3个组件：

+ CNN backbone，用于提取图像特征；
+ 编解码transformer;
+ 前馈网络FFN, 用于最终的检测预测；

#### 目标检测集预测损失(Object detection set prediction loss)

DETR设置了一个固定的参数N，推理模块最多能检测N个目标。这个参数N一般设置为远大于一般目标数量的值。训练的主要困难在于对预测结果打分，我们的损失函数先建立预测值与真值的二部图匹配关系，然后进行损失优化。
假设真值是y, 预测值是$\hat{y} = \hat{y_i}^{N}_{i = 1}$. N远大于真值目标数量，且真值存在空集的可能，通过最小代价距离找到二部图匹配关系，代价距离计算公式为：
 ![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1ty4cjUQQ2i)
此步的损失计算本质上与启发式指派规则将proposal或anchor匹配到真值相同，只不过通过二部图匹配找到一对一的匹配关系，没有重复检测。
损失函数如下：
![image.png](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/1tyFVvteJQO)
其中，当真值为空集时，将对数项降低10倍进行正负样本平衡，这个trick与Faster-RCNN中的二次采样思想相

