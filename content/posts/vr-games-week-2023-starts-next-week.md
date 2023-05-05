---
title: '2023年VR游戏周即将到来！'
date: '2022-12-27T13:06:38+08:00'
description: 'Vr Games Week 2023 Starts Next Week'
author: '虚幻引擎官网'
cover: 'https://cdn2.unrealengine.com/vr-week-2023-header-4-1920x1080-376e6c48383f.jpg?resize=1&w=1920'
tags: ["功能", "动画", "广播与实况", "虚幻引擎"]
theme: 'light'
---

# 摘要

3D检测的可靠性和准确性对很多应用都很重要，比如自动驾驶、服务机器人等。  

**MPPNet结构**:本文提出MPPNet，一种高效且灵活的3D检测框架。 该框架是一个三级架构，通过代理点云(proxy points)实现多帧点云的特征交互和编码。三个模块的主要功能分别是：

- 当前帧的特征编码；
- 短序列的特征融合；
- 全序列的特征融合；

**关键模块**：为保障全序列点云处理的资源消耗合理可控，在上述的第二级和第三级序列点云特征融合模块中，我们提出两个核心模块：

- 组内特征混合(intra-group feature mixing)
- 组间特征注意(inter-group feature attention)

**核心思路**：引入多帧间交互的代理点云(proxy points)，其作用是：保证目标在连续帧上表征的一致性及多帧特征交互的连接桥梁。

**实验结果**：WOD上SOTA   

**代码**：https://github.com/open-mmlab/OpenPCDet

# 基本介绍

**基于点云序列的3D检测方法**：最近有几篇文章证明：运用多帧连续点云可以有效提高3D检测的效果，这些文章包括：

- AFDetV2;
- CenterPoint?CenterFormer?
- Rsn: Range sparse net for efficient, accurate lidar 3d object detection

**序列检测面临的问题**——**长尾效应**：对于运动目标，多帧的点云序列会导致运动长尾现象，这种现象会影响检测精度。正因为此问题，上面提到的方法大多把序列长度控制在4帧以内，超过4帧会由于长尾效应导致检测效果下降。长尾效应的现象如下图所示：

![image-20230504141935024](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504141935024.png)

**本文的解决方法**：本文的MPPNet处理流程分两阶段：

1. **3D轨迹proposal**：第一阶段复用之前的一阶段3D检测算法，获得3D轨迹的proposal。具体是使用PoiontPillars和CenterPoint的检测器生成proposal box，然后以proposal box及速度进行关联计，生成proposal轨迹；

   该阶段的难点在于：多帧间的点云特征聚合不好做，因为不同帧之间点云的分布情况也不一样。

   为解决分布对齐及聚合问题，我们提出proxy points概念：在proposal box的固定位置和相对一直的位置生成proxy point，使多帧间的特征聚合更容易一些。

2. **多帧特征聚合**：第二阶段是本文核心，输入3D轨迹的proposal之后，按目标聚合多帧特征，进而产生更准确的检测结果。第二阶段分为3个模块：

   - 当前帧特征编码：采用PointNet++中的SA模块，使用proxy point进行编码，获得目标的几何特征；同时对帧间相对位置进行编码，获得目标的运动特征；
   - 短序列特征聚合：由于直接使用proxy point进行特征汇聚会带来很大的计算负担，因此将proposal轨迹分割成小的轨迹片段，然后用3D MLP模块进行特征聚合；
   - 全序列特征聚合：使用交叉注意力模块对上一步生成的轨迹片段特征进行进一步聚合；

通过上述3个阶段的特征聚合，我们获得了丰富的轨迹特征，这为之后产生高质量的检测框提供了很好的基础。

# 相关工作

## 基于单帧点云的3D检测方法

基于单帧的3D检测方法主要分3类：

- **基于point的方法**：代表性的方法有PointRCNN, STD, Vote3Deep， PointNet等，由于有PointNet++提出的SA等优秀算子，使得此类方法可以获得高质量的空间信息；
- **基于voxel/pillar的方法**：PIXOR，PointPillars, AfDet将3D转换为2D形式再检测；SECOND，CenterPoint, VoxelNet使用3D CNN提取3D特征；
- **结合voxel和point的方法**：PointNet++, PV-RCNN++, Lidar RCNN使用基于voxel的方法提取3D特征，使用基于point的方法进行box refine;

## 基于视频的2D检测方法

有一系列基于视频的2D检测方法，总的来说可分为两类：

- **硬关联方法**：这些方法的核心思路是使用之前帧的外观和运动特征来对齐当前帧的目标，运用的主要方法是光流、运动及LSTM来进行多帧的对齐和信息聚合。代表性方法有：
  - Object detection in videos with tubelet proposal networks
  - T-cnn: Tubelets with convolutional neural networks for object detection from videos
  - Mining inter-video proposal relations for video object detection
  - New generation deep learning for video object detection: A survey
  - Fully motion-aware network for video object detection
  - Flow-guided feature aggregation for video object detection
  - video object detection with an aligned spatial-temporal memory
  - Detect to track and track to detect
  - Object detection in video with spatiotemporal sampling networks
  - Impression network for video object detection
  - Flow-guided feature aggregation for video object detection
  - Deep feature flow for video recognition
- **基于self-attention的软关联方法**：最近几年，很多方法开始采用self-attention来进行特征对齐，代表性方法有：
  - Sequence level semantics aggregation for video object detection
  - Memory enhanced global-local aggregation for video object detection
  - Relation distillation networks for video object detection

## 基于点云序列的3D检测

- **短序列检测方法**：一些SOTA文章已经证明：使用短序列能显著提高3D检测效果，比如：
  - AfDetV2;
  - Rsn: Range sparse net for efficient, accurate lidar 3d object detection
  - CenterPoint;
- **长序列检测方法**：当序列长度更长时，上述的短序列检测方法开始出现负增益。于是出现一些如下解决方法：
  - 3D-MAN：3d-man: 3d multi-frame attention network for object detection, 采用注意力机制和memory bank来对齐多视角的时空信息；
  - SimTrack: Exploring simple 3d multi-object tracking for autonomous driving, 提出一种基于点云的3D检测跟踪一体化框架；
  - Offboard3D: Offboard 3d object detection from point cloud sequences, 提出一种离线检测方法，以过去帧和未来帧作为输入，极大地提高了检测精度；

# 整体结构

![image-20230504142014013](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504142014013.png)

## 单帧特征编码

特征编码模块主要功能是基于proposal trajectory对目标进行特征编码，主要的编码特征是几何特征和运动特征。具体的编码方式如下：

- **几何特征编码**：以proxy points进行几何特征编码, 其输入数据包括：

  - proposal box：每个proposal box有8个顶点，1个中心点，proposal box表示为

    $B^t =\{{b_j^t}\}_{j=1}^9$；

  - proposal trajectory：T帧的轨迹有m个，数学形式表示为

    $K^t =\{{l_1^t, ... , l_m^t}\}_{t=1,...,T}$；

  处理流程如下：

  1. 计算框的变化信息：当前帧的各个porposal box与其历史轨迹框的各点的距离，${\Delta}l_i^t = Concat(\{l_i^t - b_j^t\}_{j=1}^9)$；

  2. 轨迹框信息加上变化信息：将第一个的框的变化信息加入到框的信息上，得到增强的特征

     $F^t = \{{(l_i^t, \Delta_i^t)}\}_{i=1}^m$;

  3. 使用PointNet++中提出的Set Abstraction对每个proxy point聚合其领域点，计算过程为

     ${g_k}^t = SetAbstraction(p_k^t, F^t), for k = 1, ... , N$;

- **运动特征编码**：

  1. 计算proxy point与proxy box顶点、中心点的距离：${\Delta}p_k^t = Concat(\{p_k^t - b_j^t\}_{j=1}^9)$；

  2. 加入时间因素：

     ${f_k}^t = MLP(Concat({\Delta}p_k^t, e^t)), for k = 1, ... , N$;

- **单帧特征汇总**：将运动特征和几何特征结合起来，最终输出的proxy point特征是$r_k^t = g_k^t + f_k^t , for k = 1, ... , N$; 单帧中所有目标的特征是$R^t = \{{r_1}^t, ... , r_N^t\}$;

  

## 组内特征聚合

该模块负责分组后，对组内的proposal轨迹进行时间特征编码。 第i组的特征是$G^i = \{r_i^t, ... , r_N^t\}$ ,  $ G^i \in R^{T^i, N, D}$ , 其中D是每个proxy point的特征维度。

编码的方式就是MLP，公式为$\hat{G^i} = MLP^{4d}(MLP(G^i))  ,  for  s = 1, ... , S$ ;



## 组间特征聚合

该模块的功能是对所有组的proxy point进行特征聚合，来获取更丰富的信息。 该聚合功能通过cross-attention机制来实现，具体处理流程如下：

1. 总结所有轨迹信息：通过MLP层对S组、N个轨迹、D维proxy point特征进行信息汇总和总结，所有轨迹信息为H, 则公式为$\hat{H} = MLP(H), \hat{H} \in R^{N * D}$ ;
2. 为每个组的轨迹聚合全局信息：使用cross-attention机制将全局信息聚合到每个组里，公式为：$\hat{G^i} = MultiHeadAttn(Q(\hat{G^i} + PE), K(H + PE), V(H)), for  i = 1, ... , S$; 其中PE是将proxy point索引经过MLP映射得到的proxy point位置编码信息；



## 检测头

使用一个简单的transformer层来获得每组的特征向量，基本处理流程如下：

1. 使用multi head attention从每组聚合信息，公式为

   $ E^i = MultiHeadAttn(Q(E), K(\hat{G^i} + PE), V(\hat{G^i})), for  i = 1, ... , S$

2. 

## 损失函数

损失值是置信度的损失值与BOX回归框损失的和，公式为$L = L_{conf} + \alpha*L_{reg}$; 其中置信度和box损失值的计算方式与CT3D中的一样。



# 实验结果

## 数据集

WOD是一个规模很大的3D检测数据集，包含1150个序列，其中798个训练序列，202个验证序列，150个测试序列。 本模型在训练集上训练，在测试集和验证集上evaluate。

- **官方评价指标**：WOD官方的评价指标有两个：
  - mAP：3D Average Precision的均值；
  - mAPH: mAP加权Heading准确度的指标；
- **难度区分**：根据每个目标包含的point数量，数据集分成2个部分：
  - LEVLE 1：每个目标包含point数量大于5；
  - LEVEL 2：每个目标至少包含1个point；

## 实现细节

- **训练细节**：两阶段的训练是分开的：
  - 第一阶段的RPN子网络按照CenterPoint和PointPillars的训练策略进行训练；
  - 第二阶段子网络是以ADAM优化器进行6个epoch的训练，学习率为0.003， batchsize为16，针对proposal trajectory的IOU阈值为0.5；
- **超参数**：
  - 每个proposal的proxy point数量N=64；
  - 每个proposal box随机采样点的数量为128；
  - 每组的帧窗口长度为4；

## 测试结果

- **与PV-RCNN++对比**：我们的第一阶段自网络使用CenterPoint, 输入序列长度设为16， 测试结果显示：LEVEL2样本的车、行人、自行车的mAPH分别提高5.05%，8.76%，4.9%。
- **与3D-MAN对比**：LEVEL2的vehicle的mAPH提高7.82%；
- **与CT3D对比**：我们将CT3D扩展成多帧的CT3D-MF，与扩展后的模型相比，vehicle的mAPH提高4.12%；

![image-20230504142042900](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504142042900.png)

![image-20230504142059343](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504142059343.png)

# 个人分析与总结

**作者信息**：MPPNet是港中文的李鸿升团队、MPI的史少帅及华为诺亚实验室的徐航团队合作的文章，发表于2022年5月。这几位作者是基于lidar的3D检测领域的重量级人物，近几年提出了很多非常不错的3D检测方法。比如2018年的PointRCNN、2020年的PV-RCNN、Voxel RCNN、PV-RCNN++、2021年的VoTr、Pyramid RCNN、2022年的Point2Seq、2023年的DSVT等方法。

现在这篇基于点云序列的3D检测方法，是作者对SECOND结构的网络进行组件优化、组件transformer化之后的又一个尝试方向。

**创新点**:  对基于点云序列的3D检测方法进行了完善，提出了新的组件和框架，包括几何特征编码模块、运动特征编码模块、组内特征编码模块和组间特征编码模块。

具体的创新点为下图中红色框所代表的模块：

![image-20230504143232657](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230504143232657.png)

**进一步优化方向**：当前基于点云序列的3D检测方法还比较少，没有公认的成熟的方法。本文使用基于MLP和cross attention的组件进行proposal序列的特征提取和聚合，并获得了明显的提升。后续可能的演进方向有：

+ 基于MPPNet框架，提出更高效的序列特征提取和聚合机制：当前学术界对Transformer的理解还不够深刻，待Transformer发展更成熟后，基于Transformer的序列特征聚合机制也将进一步发展，也许到时候学术界会提出更好的针对点云序列的特征提取和聚合机制；
+ 与点云补全方法结合：2022年一来，涌现出几篇基于点云补齐的3D检测方法，比如BtcDet/PDA/GLENet等，可尝试将此类方法与点云序列方法结合；
+ 将序列检测机制与LC融合机制相结合：当前LC融合框架的检测指标逐渐超过只基于Lidar的3D检测指标，说明学者们逐渐提出了有效的LC融合机制。而基于点云序列的3D检测方法也被越来越多的人注意到，逐渐提出一些有效的检测框架。不难想像，后面这两种方向的3D检测方法还会继续演进。我们一方面可以不断迭代两种方向的检测方法，另一方面也可以将两种方法结合，也许会取得不错的效果，比如MPPNet+LoGoNet、MPPNet+VirConv等；