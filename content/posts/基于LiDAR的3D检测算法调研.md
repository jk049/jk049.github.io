---
title: '基于LiDAR的3D检测算法调研'
date: '2022-12-27T13:06:38+08:00'
description: '盘点2017年以来的典型3D检测算法'
author: 'jk049'
cover: 'https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 220504.png'
tags: ["Lidar", "3D"]
theme: 'light'
---

# 总体介绍

自动驾驶技术已经广泛应用与自动驾驶卡车、机器人出租车、送货机器人等领域，而自动驾驶技术最核心的功能模块就是感知系统。感知系统的输入通常是多模数据，包括来自camera的图像、来自LiDAR的点云、以及高精度地图等数据；感知系统的输出是路上关键元素的语义信息和几何信息。高精度的感知结果是下一步轨迹预测及路径规划的保障，因此3D目标检测是感知系统最重要的任务之一。

3D目标检测的目的包括：

+ 预测目标的位置；
+ 预测目标的尺寸；
+ 关键目标的分类，如行人、机动车、自行车等；

与2D目标检测相比，3D检测更注重目标的位置和真实世界中的3D坐标。通过3D检测结果的几何信息，可以获得自动驾驶车辆与感知目标的距离等关键信息。

本文介绍基于LiDAR的3D检测方法，即基于点云或range图像的3D检测方法。具体安排如下：

1. 介绍基于LiDAR的3D检测典型方法，包括：
   + 基于point的检测方法；
   + 基于voxel/pillar的检测方法；
   + 基于range图像的检测方法；
   + 基于多种技术的混合方法，包括transformer/voxel/pillar等技术、点云表征方式以及各检测框架的混合；
   + 基于对输入数据增强的方法；
2. 对基于LiDAR的3D检测算法进行分析与总结，展望下一代基于LiDAR的3D检测技术发展趋势。

基于LiDAR的3D检测方法的里程碑如下图所示：

![image-20230225204438920](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230225204438920.png)

接下来，本文将介绍上图中各种典型方法的原理与架构方面的细节。



# 典型方法

虽然当前基于图像的深度学习模型取得很多突破性的进展和质的改善，但是无法将基于图像的深度学习方法直接应用于LiDAR数据。主要原因是图像的像素分布均匀，但是点云分布稀疏且不均匀，因此对点云的特征提取模型需要特殊设计；同时，对于LiDAR的range图像而言，虽然其数据分布稠密，但是range图像呈现的是3D信息，而不是RGB信息，因此直接在range图像上使用卷积神经网络可能不会得到好的效果。

另一方面，自动驾驶场景的检测通常要求实时性，因此如何在满足实时性的同时满足高精度的检测要求仍是自动驾驶领域未攻克的难题。

基于此背景，本章节将介绍基于LiDAR的3D检测典型算法。



## Vote3Deep

该文章是牛津大学于2016年9月提出的一种基于投票机制(voite scheme)的3D检测方法，在KITTI上的car easy测试集的AP达到76.79%。2016年以前，基于LiDAR的3D检测方法大多是基于投票机制的，比如Vote3D/Voting for Voting等。

Vote3Deep是基于投票机制的3D检测方法的最后一篇代表性文章，借鉴了许多之前的基于vote机制的3D检测算法的优点。之后的LiDAR 3D方法开始转向voxel/pillars/transformer/RCNN等技术方向，同时指标也有了大幅提升，vote机制几近没落。

该文章的主要方法是将稀疏点云数据转换为不同可变尺度的grid，然后用CNN对3D框进行预测和回归。

对于每个非空grid，基于统计的方法提取其特征向量，特征向量的信息包括：是否为空、反射率的均值和方差以及长宽高的均值和方差。获得特征向量后，采用类似Voting for Voting in Online Point Cloud Object Detection

的投票机制，对特征向量进行稀疏卷积。为了避免为每个框回归具体尺寸，文章将每个类别的框都设置为固定尺寸。投票过程的示意图如下所示：

![image-20230219153018076](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219153018076.png)

基本过程是通过卷积滤波核(convolutional filter kernel)实现，用每个gird以及其周围grid的点云状况表征该点云信息。

测试指标如下：

![image-20230219153406877](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219153406877.png)



## PointNet

PointNet是Leonidas团队于2016年12月提出的一种基于LiDAR的3D处理框架，之后作者还提出了改进版本PointNet++。之所以称为3D处理框架而不是3D检测框架，是因为作者在效果验证环节，用3D目标分类数据集和语义分割数据集对框架进行效果验证，没有在当时常用的KITTI 3D检测数据集上验证。

本文在这里介绍这篇文章主要有两方面原因：

1. PointNet是第一个可以直接处理LiDAR点云的方法，之前的方法总是需要将点云转换为其他数据形式，然后才能进行推理；
2. 就目前而言，直接处理点云的3D检测方法基本都是基于PointNet的演进，之后的很多文章都使用或借鉴了PointNet中处理点云的方法；

PointNet的架构如下图所示：

![image-20230219194343137](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219194343137.png)

由上图可知，PointNet由一系列mlp、max pool、BN、ReLU等组成，具体分析不在这里展开。

由于PointNet直接处理点云数据，所以内存消耗大，处理速度慢。后面基于PointNet的演进方法主要进行两方面的优化：

+ 改变PointNet的应用场景：将PointNet应用于部分点云而不是全部点云，比如FrustumPointNet将PointNet独立应用于每个Frustum中；

+ 提取PointNet的部分层组成独立模块：如后来的VoxelNet将PointNet中某些部分提取出来，组成VFE(Voxel Feature Encoding)模块, 而再后来的SECOND又对VFE进行了优化；

  

## VoxelNet

VoxelNet是Oncel Tuzel团队于2017年11月提出的一种基于LiDAR的3D检测算法。该算法的主要贡献是首次将无规律的点云数据划分为规律的数据形式：将点云分割成相同大小的长方体，该长方体称为voxel。之后很多研究都沿用了这种称为voxelization的点云转换方法，比如SECOND、PV-RCNN、Voxel R-CNN、VoTr等。voxel的示意图如下所示：

![image-20230219205801444](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219205801444.png)

文章中，每个voxel的长宽高尺寸是(0.2m, 0.2m, 0.4m)。

VoxelNet的整体架构如下图所示：

![image-20230219210232313](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219210232313.png)

具体来说，VoxelNet将输入点云转换为相同大小的3D voxel，然后用VFE(voxel feature encoding)层将每个voxel内的点云转换为特征表示；最后用RPN进行3D框的检测和回归。其中，VFE的借鉴了PointNet的结构，特点也是处理速度较慢，后续工作如SECOND提出了效率更高的VFE层。

作者没在KITTI测试集上对此方法进行验证，只在验证集上进行了验证，对car easy样本的AP指标为81.97%，详细数据如下：

![image-20230219205415319](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219205415319.png)

【注】后续的PointPillars在横向指标对比中，说VoxelNet在KITTI测试集上对car easy样本的AP为77.47%。



## SECOND

SECOND是李博团队于2018年8月提出了针对LiDAR 3D检测任务的one stage检测方法。该方法是基于VoxelNet的演进，其核心贡献是首次引入针对点云的稀疏卷积，同时对VoxelNet的VFE层进行了速度优化。SECOND框架的特点是AP高的同时速度快，在1080Ti上每帧的推理速度为50ms，而之前的VoxelNet需要200ms。

SECOND的整体架构如下图所示：

![image-20230219213834010](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219213834010.png)

可见该框架与VoxelNet的结构非常相似：voxelization之后进行VFE，然后接3D卷积层，之后RPN。不同之处在于SECOND对VFE进行了速度方面的优化，同时借鉴空间稀疏卷积(spatially sparse convolution)和submanifold convolution的思路，将VoxelNet中的传统3D卷积层换为稀疏卷积层。

测试指标：在KITTI测试集上，对car easy样本的AP达到83.13%，详细数据如下：

![image-20230219214758497](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219214758497.png)



## PointPillars

PointPillars是提出于2018年12月的一种基于LiDAR的3D检测算法。该算法的特点是非常快，在1080Ti上每帧点云的推理速度达到60-100Hz。PointPillars的核心贡献是提出另一种类似voxel的点云规则化的方法：将无规律的点云转换为规律的柱形长方体，该长方体称为pillar。pillar示意图如下所示：

![image-20230219220115986](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219220115986.png)

论文中，每个pillar的长和宽都是0.16m，高度为整个点云高度，对应KITTI数据集则为4m。

PointPillars的整体架构如下图所示:

![image-20230219220712381](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219220712381.png)

该框架主要包含3个模块:

+  特征编码模块: 将点云pillar化后进行特征提取，然后转换为稀疏伪图像;
+  2D卷积backbone: 对伪图像进行特征提取;
+  检测头: 回归最后的3D框, 使用SSD作为head.

测试指标：在KITTI测试集上，对car easy样本的AP达到79.05，不及前面的SECOND，但是在中等和困难样本上的AP比SECOND高。详细数据如下：

![image-20230219221500708](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230219221500708.png)



## IPOD

IPOD是贾佳亚团队于2018年12月提出的一种3D检测算法，该算法的核心思想是从点云直接生成proposal。不同于对点云进行各种转换后的3D检测, IPOD直接基于每个点播撒proposal种子, 然后生成proposal, 然后对proposal进行特征提取, 最后生成检测框。

IPOD的整体框架如下图所示：

![image-20230220221917115](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220221917115.png)

基本处理流程为:

1. 先从图像生成目标的语义分割;
2. 将图像上的语义分割结果映射到点云中;
3. 在点云上产生proposal;

笔者认为上述第2步，将语义分割信息映射到点云的过程会引入偏移问题，这种映射偏移问题在LC融合算法中较常见。也正是因为这种映射便宜问题难以解决，最近的LC融合算法都改向了基于Transformer的软关联方式，抛弃了早期的硬关联等映射机制。可能正是由于硬关联导致的映射偏移问题，才导致该算法在指标上不出彩吧。

从指标上看，该算法对car easy的AP指标为79.75，远低于18年8月提出的SECOND；另一方面，作者宣称该算法对中等和困难场景的检测效果较好，但实际上，同期的PointPillars在困难场景的AP指标更高。详细数据如下所示：

![image-20230220222242442](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220222242442.png)



## PointRCNN

PointRCNN是李鸿升/王晓刚团队于2018年12月提出的基于LiDAR的3D检测算法。该算法是两阶段算法，第一阶段是自底向上生成3D proposal，第二阶段是由proposal获得预测结果。该方法与IPOD类似，都不需要对点云进行格式转换，直接处理点云来生成proposal。

PointRCNN的整体结构如下图所示：

![image-20230220224102322](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220224102322.png)

该算法包含如下几个技术点：

+ 以PointNet++为3D Backbone；
+ 用focal loss来改善样本不均衡问题；
+ 以3D框不会重叠为重要假设，将点云分为前景点和背景点，然后以自底向上的方式从前景点生成高质量的proposal；



测试指标：在KITTI测试集上对car easy样本的AP为85.94%，同时对中等和困难样本的检测指标也有全面提升。具体数据如下：

![image-20230220225023503](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220225023503.png)



## PIXOR

PIXOR是Raquel团队于2019年2月提出的一种基于LiDAR的一阶段3D检测算法。该算法的特点是速度很快，介于SECOND和PointPillars之间，在1080Ti上的推理速度约为28fps。

PIXOR的整体架构如下图所示：

![image-20230220225722866](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220225722866.png)

其中：

+ 输入数据转换为BEV，额外增加一个通道携带高度信息，同时丢弃反射强度信息；
+ PIXOR检测器由一系列残差块、上采样以及3*3的卷积层组成；

检测指标：作者在测试指标方面表现很奇怪，没有对3D框的准确性进行验证，只公布了BEV框的测试指标。具体数据如下：

![image-20230220230310084](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220230310084.png)

对于KITTI的car easy样本，BEV检测框的AP只有81.70%，可以推测其3D的的AP指标更低，与之前的SECOND相比毫无竞争力。



## STD

STD是贾佳亚团队于2019年7月提出的一种基于LiDAR的3D检测算法，该算法的特点是对困难样本的检测指标提升明显。

该算法是两阶段检测算法：

+ 第一阶段是自底向上的proposal生成网络, 以原始点云为输入, 在每个点撒proposal种子, 然后生成proposal; 然后用PointsPool提取每个proposal的特征;
+ 第二阶段使用并行IOU来加速获取3D box;

该算法的整体框架如下图所示：

![image-20230220231813153](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220231813153.png)

该算法生成proposal的方式似乎结合了IPOD和PointRCNN的部分理念。为了生成精确的基于point的proposal，文章使用球形anchor，并且给每个anchor赋一个标签。
对每个proposal，使用新的PointPool层将点云特征从稀疏表示转换为稠密表示，最后接一个框预测网络。

另外一个核心贡献是使用PointPool对proposal进行特征提取。具体分为3步：

1. 每个proposal内随机选N个点，其坐标及语义特征为初始特征；
2. 使用voxelization层将每个proposal进一步划分成多个voxel;
3. 使用VFE(voxle feature encoding)层对每个voxel进行特征提取；

测试指标：在KITTI测试集上，对car easy样本的AP为86.61%， 增长一般，但是对困难样本的AP达到76.06%，增长非常大。具体数据如下：

![image-20230220232138171](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230220232138171.png)



## 3DSSD

3DSSD是贾佳亚团队继STD之后提出的另一个基于LiDAR的3D检测算法，于2020年2月提出。该方法最大的创新在于：3DSSD是第一篇直接处理点云的一阶段检测算法。结果方面，检测精度和速度取得了不错的平衡：在Titan V上的推理时间为38ms，AP指标于2阶段方法相当，对car easy样本的AP为88.36%。不过该AP指标不是很可靠，稍后进行说明。

3DSSD的整体框架如下图所示：

![image-20230221220302915](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221220302915.png)

框架基本由3部分组成：

+ 直接处理点云的3D Backbone;
+ CG(candidate generation)层：主要功能是点云特征提取及下采样、偏移等；
+ 检测头；

测试指标：对car easy样本的AP为88.36%，但是其他方法的指标虚高，主要原因应该是作者使用了4种数据增强方法。如果不进行数据增强，该算法的AP指标可能会下降2%左右。详细数据如下：

![image-20230221221557056](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221221557056.png)



## CenterPoint

CenterPoint是Philipp Krähenbühl团队于2020年6月提出的一种基于point的3D检测方法，是继PointNet系列和3DSSD之后，另一篇基于point的代表性检测算法，整体框架是基于CenterNet结构在点云上的应用。

值得说明的是，Waymo数据集于2020年发布，该文章率先在Waymo上进行了指标测试，开启了基于Waymo的3D检测时代。Waymo与KITTI不同点在于：Waymo数据集更复杂，每帧点云数量更大，许多在KITTI上表现较好的方法，在Waymo上的测试指标反而一般。主要原因就是那些方法的显存消耗大，Waymo上的大点云数量限制了那些方法的效果。因此后来的很多方法都致力于减少显存消耗方面的优化。

CenterPoint的整体框架如下图所示：

![image-20230221224950802](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221224950802.png)

1. 第一阶段是head模块, 预测各种信息, 包括类别, 尺寸, 位置, 朝向, 速度等信息. 其中:
   + 目标中心点是通过热力图来估计的. 具体参见CenterNet;
   + 位置细化回归头: 
   + 速度头: 速度用于跟踪;
2. 第二阶段从backbone的输出提取附加的点特征, 即提取每个面的中心点特征. 由于在俯视图上重合, 只需提取5个点的特征, 具体操作通过MLP实现. 

测试指标：在Waymo上机动车L1样本的mAP为80.2，没在KITTI上测试。具体数据如下：

![image-20230221225629627](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221225629627.png)



## PV-RCNN

PV-RCNN是李鸿升、王晓刚团队于2020年6月提出的一篇基于LiDAR的3D检测算法。该算法的主要贡献在于：首次提出结合point和voxel各自优点的方法。之前的方法要么基于point，要么基于voxel/pillars：

+ point保留了精确的深度、距离信息，但是基于point的方法计算量大，内存消耗多；
+ voxel计算量少，能产生高质量的proposal，有更多的上下文信息，但是voxelization过程种会丢失深度和距离等信息；

PV-RCNN对之前的算法进行了细致的分析，然后创造性地提出一种将point和voxel优点结合的方法。实验结果也表明，PV-RCNN这种结合两种数据形式的思路是正确的，而且作者提出的结合方式也很有效：从指标来看，在KITTI上的car easy样本的AP达到90.25%，远远高于之前的方法。而该方法在相当长的一段时间内，在KITTI上的保持着非常有竞争力的指标。直到2021年6月，随着SE-SSD的提出，KITTI上才有了比PV-RCNN有明显优势的新指标。而这段时间内，各种方法都在借鉴PV-RCNN的设计思路，企图对指标有进一步的刷新，比如Voxel RCNN/PV-RCNN++/M3DeTR等优秀算法。

PV-RCNN算法的缺点是计算速度慢，内存消耗多，尤其是在Waymo等点云数量大的数据集上，可以将PV-RCNN的缺点进一步放大。

PV-RCNN的设计思路是：基于团队之前提出的PointRCNN框架进行优化，加入voxel的信息。整体框架如下图所示：

![image-20230221231843373](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221231843373.png)

该算法的核心模块有：

+ voxel-to-keypoint 编码: 使用voxel CNN对voxel进行稀疏3D卷积来获得voxel特征和高质量proposal; 为减少此步骤的内存开销, 选择少量关键点来概括voxel的全局3D信息; 其中关键点特征由邻域voxel特征聚合而成, 而领域voxel特征由PointNet学习获得. 此步骤可以通过多尺度特征高效获得全局信息.

+ keypoint-to-grid ROI特征抽象: 提出一种ROI-grid pooling模型, 使用多尺度的关键点特征来聚合grid点的特征. 此步骤对置信度估计和3D框位置优化有帮助.

具体细节如下:

1. 3D voxel CNN: 将输入点云划分为$L*W*H$的相同大小的方块,  然后计算非空voxel的特征, 每个voxel的特征为其内部所有point的特征的均值. 具体计算时使用$3*3*3$的3D稀疏卷积.
2. 3D proposal生成: 先将3D特征图按Z轴方向切分, 堆叠成一系列的2D特征图, 然后借鉴SECOND和PointPillars的anchor方法生成proposal, 该方法生成的proposal的召回率比PointRCNN和STD的高, 如下表所示:
   ![image.png](https://s.readpaper.com/T/1wVYltKRcLC)
   原因在于: 传统3D Voxel CNN提取的特征下采样太多, 丢失了准确的位置信息, 即使上采样也只能获取很少的邻域特征, 所以进一步的proposal的召回率就不够高. 而PointNet方法可以获得任何范围的邻域特征, 所以其特征编码能力远强于voxel CNN方法.
3. 为结合voxel和point方法, 本文提出RoI grid pooling模块, 具体包含两个技术点:
   + voxle-to-keypoint: 从voxel中采样2048或4096个点作为关键点, 然后用voxel set abstraction模块对关键点进行特征编码, 此处借鉴PointNet++的集合抽象方法; 然后用关键点生成proposal, 此处有个关键点权重计算模块, 前景关键点和背景关键点的重要性不同, 该权重计算模块为PKW(predicted keypoint weighting), 结构如下图所示:
     ![image.png](https://s.readpaper.com/T/1wVh6ufzZz6)
   + keypoint-to-grid: 从每个proposal中采样$6*6*6$个grid point, 从关键点聚合grid point特征时, 设置半径r, 每个grid point从r范围内的关键点聚合特征,. 我们设置不同大小的r来获得不同感受野的特征. 之后用两层MLP获得proposal的特征表示. 示意图如下所示:
     ![image.png](https://s.readpaper.com/T/1wVjudn7dvU)
     4, proposal细化及置信度预测: 用两层MLP实现.

测试指标：

![image-20230221232334359](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221232334359.png)

![image-20230221232431908](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221232431908.png)



## Pointformer

Pointformer是黄高、Li Erran Li团队于2020年12月提出的基于LiDAR的3D检测算法。该算法首次以Transformer作为点云特征提取的3D Backbone，从此开启了基于transformer的LiDAR 3D检测时代。后面借鉴该transformer思路的典型算法有：M3DeTR/CT3D/VoTr/CenterFormer/DSVT等。

Pointformer的整体结构如下图所示：

![image-20230221233334200](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221233334200.png)

其中包含以下几个组件:

+ LT(local transformer)模块: 学习目标级的上下文依赖区域特征, 代表point与局部区域的信息交互;
+ 坐标细化模块: 调整从FPS(furthest point sampling)采样获得的目标中心点, 来改善proposal的质量;
+ LGT(Local-Global Transformer)模块: 从更高层面的表征上结合局部特征和全局特征;
+ GT(Global Transformer)模块: 在场景层面学习注意上下文的特征;

测试指标：在KITTI上对car easy样本的AP为87.13%，除了PV-RCNN，与其他算法相比还是很有竞争性的，可惜PV-RCNN太过优秀。。。。

具体测试数据如下：

![image-20230221233523456](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230221233523456.png)



## Voxel R-CNN

Voxel R-CNN是张燕咏、李厚强团队于2020年12月提出的基于LiDAR的两阶段3D检测算法。该算法借鉴了SECOND和PV-RCNN的思路，对PV-RCNN中Point那一路做了精简，通过Voxel RoI pooling模块结合2D BEV信息和3D voxel信息，达到与PV-RCNN相似的效果。

Voxel RCNN的整体框架如下图所示：

![image-20230222183749503](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222183749503.png)

相比于PV-RCNN，Voxel RCNN砍掉了复杂耗时的关键点采样、SA层，所以运算速度和内存占用等方面都有显著改善，而这些改善也可以从指标看出。

测试指标：在KITTI上对car easy样本的AP为90.90%，比PV-RCNN提高0.65个百分点，改善不多；但是在Waymo上对vehicle L1样本的mAP为75.59，比PV-RCNN高了5.29个点，这点正是因为Voxel RCNN减少了内存消耗，进而释放了模型的能力。详细数据如下所示：

![image-20230222184635794](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222184635794.png)

![image-20230222184803155](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222184803155.png)



## PV-RCNN++

PV-RCNN++是李鸿升、王晓刚团队继PV-RCNN之后的改进版本，于2021年1月提出。该算法的主要改进方向同Voxel-RCNN一样：提高PV-RCNN结构在点云数量多的输入下的检测指标和检测速度，因此该算法主要在Waymo上进行了测试。

PV-RCNN++的整体结构如下图所示：

![image-20230222185617656](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222185617656.png)

 改善的主要原因有两个:

+ 基于sector的proposal质心采样可以产生更具代表性的keypoints；
+ VectorPool可以更好聚合局部point特征, 同时减少资源消耗;

测试指标：

![image-20230222191034447](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222191034447.png)



## M3DeTR

M3DeTR是Larry Davis, Dinesh Manocha团队于2021年4月提出的一种3D检测算法。该算法的创新点在于：提出一种3D检测框架，可以将Point、voxel的数据形式与Transformer结合起来。该框架结构对多模传感器融合很友好，基于LC融合的3D检测算法可以考虑借鉴此思路。

该算法的基本框架如下图所示：

![image-20230222203803579](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222203803579.png)

基本模块如下:

1. 多表征形式的特征结合: 将输入点云进行编码到3个内嵌空间, 即voxel空间, point空间和BEV空间. 编码过程如下:
   + voxel编码: 通过使用VoxenNet中的voxelization层将点云划分为voxel, 然后使用3D稀疏卷积提取不同尺度的voxel特征, 具体提取的特征为$f^{voxel} = \{f^{voxel  1*}, f^{voxel  2*}, f^{voxel  4*}, f^{voxel  8*}\}$, 这点于VoxelNet和SECOND不同;
   + BEV编码: 对voxel的8倍下采样特征使用2D卷积, 然后生成BEV形式的特征. 具体过程是将z轴信息channel化, 然后2D卷积包含两个encoder-decoder模块, 每个encoder-decoder模块包含用来生成top-down特征的2D下采样卷积层,和上采样到输入尺寸的反卷积层; 其中不论encoder还是decoder, 都包含一系列的$3*3$卷积层, BN层和ReLU层;
   + 每帧点云数据都有10K以上的point, 为了确保内存消耗和信息有效性, 我们采用最远点采样FPS(Furthest Point Sampling)算法对点云采样n个关键点(keypoint), 采用PointNet++的SA(Set Abstration)和PV-RCNN的VSA(Voxel Set Abstration)分别产生关键点的特征; 
2. 针对多表征形式, 多尺度, 相互关系的Transformer: 分别产生$F^{voxel}, F^{point}, F^{bev}$之后, 就可以将各种表征形式的特征进一步结合, 产生多表征形式, 多尺度, 多点的特征.  Transformer基本: Transformer是一系列的encoder-decoder层,  靠自注意机制计算输入和输出的表征, 自注意的输出公式如下:
   ![image.png](https://s.readpaper.com/T/1wpsRc02dxh), 提出两层encoder的M3 Transformer, 分别是多表征多尺度Transformer和多关联关系Transformer, 如下图所示:
   ![image.png](https://s.readpaper.com/T/1wpt9icRMzR)
   + 多表征形式多尺度Transformer(multi-representation and multi-scale transformer): 对于每个点云, 对transformer层的输入序列是其一系列点云特征, 包括不同尺寸和不同表征形式的特征. 具体来说, 输入序列为$F = [F^{voxel 1*}, F^{voxle 2*}, F^{volex 4*}, F^{voxel 8*},  F^{point}, F^{bev}]$,  其输出是各形式各尺度的特征聚合之后的特征向量; 为解决各特征维度不同的问题, 我们使用单层感知机对特征维度进行对齐. 维度对齐后的特征为: $\hat{F} = [\hat{F}^{voxel 1*}, \hat{F}^{voxle 2*}, \hat{F}^{volex 4*}, \hat{F}^{voxel 8*},  \hat{F}^{point}, \hat{F}^{bev}]$, 之后将这些特征输入到transformer层中, 产生自注意特征.
   + 多关联Transformer(Mutual-relation transformer): 受PointNet/PointNet++/PointFormer启发, 我们在transformer中加入Mutual-relation层将point邻域信息融合到特征中. 具体的做法是: 将上一级的Transformer输出特征级联, 即$T = concat[T^{voxel 1*}, T^{voxle 2*}, T^{volex 4*}, T^{voxel 8*},  T^{point}, T^{bev}]$, n个关键点的级联特征$T_i$输入到Mutual-relation Transformer中, 利用多头主力机制处理后的特征即为带有点云间关联信息的特征;
3. 检测头: 两阶段, 包括RPN和RCNN, 具体参考PV-RCNN. 其中RPN用2D卷积从BEV特征中生成3D proposal; RCNN细化proposal.

测试指标：在KITTI和Waymo上都有测试，都比PV-RCNN有小幅提升。

![image-20230222204222830](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222204222830.png)

![image-20230222204113964](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222204113964.png)



## SE-SSD

SE-SSD当前在KITTI 3D检测榜上排名第2，是傅志荣团队于2021年6月提出的一种基于LiDAR的one-stage 3D检测算法。该算法的特点是检测精度高的同时速度快，在Titan XP上的单帧推理时间为30ms。该算法的核心理念是: 通过对自定义约束的软目标和硬目标来联合优化模型, 同时不引入额外计算量。

具体来说, SE-SSD包含一个teacher SSD和一个student SSD, 其中对teacher SSD进行高效的基于IOU的匹配策略来过滤器软目标, 然后通过一致损失来将Teacher SSD的预测结果与student SSD的预测结果对齐。
同时, 为使蒸馏知识最大化, 我们设计了一种形状感知的增强样本来训练student SSD, 使其预测的目标形状更准确。
最后, 为更好处理难例, 作者设计了ODIoU损失函数来监督Student SSD, 该损失函数对目标质心和朝向的预测结果有约束。 

SE-SSD的整体框架如下图所示：

![image-20230222205322092](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222205322092.png)

基本推理流程如下:

1. 先用预训练的SSD初始化teacher和student;
2. 用输入点云训练整个模型, 走两个分支:
   + teacher从输入点云中预测出相对准确的结果, 然后对这些结果进行全局转换, 将转换后的结果作为软目标, 这些软目标用来监督student;
   + 第二路也用teacher检测和全局转换, 转换之后进行形状数据增强, 然后将增强的结果输入到student中, 并对其训练, 然后对teacher和student的预测结果通过一致损失进行对齐, 同时用形状增强的student输入对student输出进行监督, 监督的损失为考虑朝向信息的IOU。

其中teacher SSD和student SSD的结构与CIA-SSD基本一致, 除了移除了置信度函数和DI-NMS。此外, 框架还包含稀疏卷积网络, BEV卷积网络, 多任务head。

测试指标：

![image-20230222205838409](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222205838409.png)

![image-20230222205632450](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222205632450.png)



## CT3D

CT3D是阿里达摩院的华先胜团队与浙大合作的一篇LiDAR 3D检测文章，于2021年8月发表。这是一种two-stage检测框架，论文重点聚焦于提高proposal的质量。核心技术点是： 对每个proposal的point提取特征时, 既进行channel-wise的上下文聚合, 也进行proposal-aware embedding。但是在数据集上的测试数据表明，该技术效果一般。因此本文不描述其技术细节，介绍此文目的在于：为评估阿里在LiDAR感知方面的能力提供参考。

CT3D整体框架如下图所示：

![image-20230222211612586](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222211612586.png)

测试指标：在KITTI和Waymo上都没什么竞争力，具体数据如下：

![image-20230222211805000](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222211805000.png)

![image-20230222211832972](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222211832972.png)



## VoTr

接下来介绍3篇我司与高校合作的LiDAR检测算法。

VoTr(VoxelTransformer)是2012实验室的徐航团队与港中文、新加坡国立大学及中山大学的梁晓丹团队合作提出的一篇3D检测算法，于2021年9月提出。该算法的核心思想如同标题：将Transformer机制与voxel结合，是一种应用于LiDAR点云的3D Backbone。

VoTr基于Transformer的架构, 使得可以通过self-attention机制获得大范围的关系信息。
由于实际情况下, 非空voxel的分布稀疏且数量巨大, 直接对voxel使用transformer会带来很大的计算量。因此, 作者提出稀疏voxle模块及submanifold voxel模块, 可以在voxel和空voxel上高效执行。

为进一步增加注意力范围的同时维持与传统方法相当的计算量, 作者为多头注意力机制提出两种注意力模块:

+ 局部注意(Local Attention):
+ 扩张注意(Dilated Attention): 

之后还提出了快速voxel查询(Fast Voxel Query)机制, 来加速多头注意力的查询过程。

VoTr整体框架如下图所示：

![image-20230222214128887](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222214128887.png)

测试指标：在KITTI上对car easy样本的AP为89.90%，在Waymo上对vehicle L1样本的mAP为74.95。详细数据如下：

![image-20230222214605825](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222214605825.png)

![image-20230222214646873](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222214646873.png)



## Pyramid R-CNN

Pyramid RCNN是我司2012实验室徐航团队与港中文及中山大学梁晓丹团队合作的一种基于LiDAR的3D检测算法。从名字不难看出，该算法是two-stage算法，而该算法致力于改善第二阶段。该算法的特点是：对简单场景的提升不明显，对中等和困难场景有较大提升。

之前的两阶段方法中, 第二阶段主要是从point或voxel中提取ROI 特征, 但是这些方法对稀疏不规则分布的点云处理效率不高, 进而导致检测结果不准确. 
为解决这个问题, 作者提出一个新的第二阶段处理模块: pyramid RoI head, 该模块可以自适应从稀疏点云中学习RoI特征. pyramid RoI head包含3个关键组件:

+ RoI-grid Pyramid: 以金字塔方式为每个RoI搜集point;
+ RoI-grid Attention: 通过结合传统的注意力机制和基于图的点云算子, 从点云中编码更丰富的信息; 
+ Density-Aware Radius Prediction: 可以动态调整RoI的rang范围, 来适应各种密度的点云输入;

Pyramid RCNN的整体框架如下图所示：

![image-20230222215326016](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222215326016.png)

测试指标：

![image-20230222215518427](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222215518427.png)

![image-20230222215426939](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222215426939.png)



## Point2Seq

Point2Seq是2012实验室徐航团队与新加披国立大学Xinchao Wang团队及港中文王晓刚团队合作的3D检测算法。该算法极富想象力，核心思想是把3D检测任务看作单词序列解码任务。Point2Seq的整体框架如下图所示：

![image-20230222220317221](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222220317221.png)

该框架与以往框架的不同点在于:
以往的框架都是一次检测所有目标, 但是此算法对各目标之间的依赖关系进行了建模, 通过各目标之间的关系信息来提高检测精度. 具体流程如下:

1. 此算法将每个3D目标视为单词序列, 然后就可以将3D检测任务转化为以自回归(auto-regressive)方式对3D场景进行单词序列解码工作;
2. 基于上述任务, 作者开发出了一个轻量化的场景到单词序列的解码模块, 该解码模块根据场景信息和之前的单词序列线索进行自动回归生成单词序列;
3. 最终预测的单词序列是对3D目标的一系列完整描述,  然后根据相似度将检测结果和真值进行映射;

测试指标：

![image-20230222220537765](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222220537765.png)

值得说明的是：该文章在新数据集ONCE上也进行了测试，而ONCE数据集也是徐航团队构建的。具体测试数据如下：

![image-20230222220659430](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222220659430.png)



## DSVT

DSVT是北大王立威团队于2023年1月提出的LiDAR 3D检测算法，目前在Waymo榜单上排名第一。该算法的特点是精度高且易部署，因为算法没用任何自定义算子。

为了能高效并行处理稀疏点云数据, 作者提出动态稀疏窗口注意机制DSWA(Dynamic Sparse Window Attention),  根据区域的稀疏性将各区域分配到每个窗口中, 然后以并行的方式计算所有区域的特征。为有效下采样和更好编码几何信息, 作者提出注意力风格的3D池化模块, 该模块很强, 没用任何自定义的CUDA算子。DSVT的整体架构如下图所示：

![image-20230222221247693](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222221247693.png)

基本处理流程如下:

1. 将输入点云转换为voxel: 使用voxel特征编码模块VFE(Voxel Feature Encoding), 在这一步, 考虑到transformer结构的感受野较大, 同时室外目标相对较小的情况, 本文不需要使用hierarchical 的表示形式, 只需一个单步长的网络, 与文章Embracing Single Stride 3D Object Detector with Sparse Transformer一致,;
2. 然后用一系列带有DSVT(Dynamic Sparse Voxel Transformer)的voxel transformer模块处理这些voxel;
3. 为了将稀疏voxel联系起来, 我们设计了两个分配方法:
   + rotated set: 窗间特征聚合;
   + hybrid window: 窗内特征聚合; 
4. 可学习3D pooling模块: 用于高效下采样, 编码精确的3D几何信息且不需要自定义算子;
5. DSTV输出的voxel特征映射到BEV;
6. 采用CenterNet样式的检测头, 检测最终的box;

测试指标：

![image-20230222221442491](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222221442491.png)



## SPG

接下来4篇文章通过改善输入数据来提高检测质量，下面一一介绍。

SPG是Waymo的Dragomir Anguelov团队于2021年8月提出的一种基于LiDAR的3D检测算法。该算法的核心思想是通过补全点云来提升检测指标，因此对中等和困难样本的检测提升显著。

作者注意到远处点云密度显著下降, 进而影响检测精度, 因此提出语义点云生成SPG(Semantic Point Generation)的方法, 用于对传统lidar检测器进行增强。具体做法是: 在预测的前景区域中产生语义点云, 恢复由遮挡/低反射率/天气等原因而丢失的点云。SPG整体框架如下图所示：

![image-20230222224525551](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222224525551.png)

点云生成的效果对比如下图所示：

![image-20230222224625440](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222224625440.png)

测试指标：

![image-20230222224830274](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222224830274.png)



## BtcDet

BtcDet是USC于2021年12月提出的基于LiDAR的3D检测算法。

实际环境中, 因为存在遮挡、反射、天气等原因, 使得lidar不能完全感知目标形状, 因此也给3D检测带来很多挑战。为解决此问题, 作者提出一种新的基于点云的3D检测方法: BtcDet(Behind the Curtain Detector), 通过学习目标的先验形状和估计倍遮挡目标的完整形状来改善检测效果。该算法的核心思想是: 模型预测目标存在的概率, 进而判断目标的形状。通过这个概率图(probability map), 本模型能产生高质量的proposal, 最后在proposal细化的时候也参考概率图的信息。

BtcDet的整体框架如下图所示：

![image-20230222225421596](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222225421596.png)

基本流程如下:

1. 首先识别遮挡和信号丢失区域, 然后用形状遮挡网络估计目标形状可能性: 

   + 对于车和自行车, 将点云按中心面对称; 
   + 以相邻目标进行点云补全;
   + 如果是球坐标系, 对被遮挡后的空间进行voxel化, 对可能存在目标的区域补点云;
   + 开始训练:

   示意图如下所示:

   ![image-20230222225538833](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222225538833.png)

2. 用backbone提取点云3D特征, 然后将3D特征和形状可能性张量结合, 将结合特征送到RPN产生proposal;

3. proposal细化;

测试指标：

![image-20230222225626081](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222225626081.png)

![image-20230222225652806](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222225652806.png)



## PDA

PDA是Steven Waslander团队于2022年3月提出的基于LiDAR的3D检测方法。该方法的特点是考虑了点云密度与距离的关系，进行针对此特性对点云进行了优化。

PDA整体结构如下图所示：

![image-20230222230820589](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222230820589.png)

该框架的基本处理流程是:

1. 从3D稀疏卷积backbone中通过voxel质心来高效获取voxel位置特征;
2. 然后将voxel位置特征通过density-aware RoI grid pooling模块利用KDE(kernel density estimation)和点云密度位置编码的自注意聚合;
3. 最后利用距离与点云密度的关系, 对3D框进行最后的细化和置信度修正;

测试指标：

![image-20230222231009796](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222231009796.png)

![image-20230222231032294](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222231032294.png)



## GLENet

GLENet是香港城市大学于2022年7月提出的LiDAR 3D检测算法，目前在KITTI榜单上排名第一。该算法的主要创新点是考虑了训练数据的准确却性，并提出改善方法。其特点是即插即用，插件化特性；同时检测精度很好。

对检测结果准确度建模的方法有两种, 如下图所示:
![image-20230222231708151](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222231708151.png)

作者采用第二种方法的设计理念,  自定义一种强有力的基于深度学习的标签不确定性量化框架。
GLENet就是获取目标框潜在位置的方法。在推理阶段, 我们生成多个潜在的目标框, 如下图所示:
![image-20230222231814207](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222231814207.png)
此外, 作者还提出考虑不确定性的检测质量估计模块UAQE(uncertainty-aware quality estimator), 对检测质量进行估计。

GLENet的整体框架如下图所示：

![image-20230222232157410](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222232157410.png)

测试指标：在KITTI测试集上对car easy样本的AP为91.67%，详细数据如下：

![image-20230222232227339](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230222232227339.png)



# 总结与展望

前面对基于LiDAR的3D检测算法进行了总体介绍及典型算法介绍。本节对前面的经典方法进行分类和总结，具体安排如下：

1. 从数据形式的角度介绍和分析基于LiDAR数据的3D检测模型，包括：
   + **基于原始点云的方法**；
   + **基于将点云划分为各种grid的方法**；
   + **基于point-voxel的方法**；
   + **基于range的方法**；
2. 从学习方式的角度介绍各种3D检测模型，包括：
   + **基于anchor的方法**；
   + **anchor-free的框架**；
   + **基于lidar的3D检测中的一些辅助任务**；

基于lidar的3D检测方法的里程碑如下图所示：

![image-20230225204505026](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230225204505026.png)



## 基于原始点云的3D检测方法

基于点云的3D检测框架如下：

1. 点云数据先经过基于点的backbone网络，点云算子对点逐步采样并学习特征；
2. 基于下采样点和特征进行3D框预测；

框架示意图如下所示：
![image.png](https://s.readpaper.com/T/1rCD8Ao2aLA)

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
![image.png](https://s.readpaper.com/T/1rMSfyaXBK4)

### 基于点云的3D检测方法总结

基于点云的3D检测方法效果主要受三个方面制约：

1. 上下文点(context point)的数量：增加上下文点的数量会获得更多的信息，但是会占用更多存储空间；

2. 特征学习过程中设置的上下文查询半径：查询半径过小会倒是上下文信息不足，半径过大会导致细粒度信息缺失；

   上下文点的数量和查询半径是需要仔细权衡的两个关键因素，这两个因素会显著影响检测模型的效率和精度。

3. 点云采样方法是基于点云的3D检测算法的瓶颈，不同的采样方法特点不同：

   + 随机均匀采样可以高效并行进行，但是会导致近处的点云过采样，远处的点云欠采样；
   + FPS(Furthest point sampling)在近远点的采样平衡性会更好，但是无法并发进行，不适合实时场景；



## 基于将点云划分为grid的方法

基于grid的数据表示形式有3种：分别是voxel, pillars, BEV。下面分别介绍各类方法：

###  1. voxels:

voxel是3D立方体，每个立方体内包含点云。通过voxelization可以很容易将点云转换为voxels。因为点云分布是稀疏的，所以很多voxel里面是空的，没有任何点云。实际应用种，只会存储非空的voxel，然后从非空的voxel中提取特征。首次提出voxel概念的文章：VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection([Oncel Tuzel](https://www.aminer.cn/profile/oncel-tuzel/53f47b73dabfaee4dc89e681),  cite2000, 2017)

类似voxel表示形式的文章：

+ Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection(Yu Gang, cite0, 2019)
+ AFDet: Anchor Free One Stage 3D Object Detection(Li Huang, cite60, 2020)
+ Center-based 3D Object Detection and Tracking([Philipp Krähenbühl](https://www.aminer.cn/profile/philipp-kr-henb-hl/53f4589ddabfaedd74e36b9a), cite350, 2021)
+ Object DGCNN: 3D Object Detection using Dynamic Graphs(Justin Solomon, cite20, 2021)
+ CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud(Chi-Wing Fu, cite100, 2021)
+ Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection(Yanyong Zhang, cite200, 2021)
+ From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network([Hongsheng Li](https://www.aminer.cn/profile/hongsheng-li/53f46582dabfaee2a1dab2ad), cite400, 2021)

此外还有多视角的voxel方法：从range视图、圆柱视图、球状视图、感知视图、BEV视图等多种视图动态voxelization和融合的机制来生成voxel，代表性方法有：

+ End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds([Anguelov Dragomir](https://www.aminer.cn/profile/anguelov-dragomir/53f43947dabfaedce554a66b)， cite200, 2019)
+ Every View Counts: Cross-View Consistency in 3D Object Detection with Hybrid-Cylindrical-Spherical Voxelization(Qi Chen, cite40, 2020)
+ VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention(Kui jia, cite10, 2022)

还有多尺度的voxel方法：voxel的尺寸不同，代表文章有：

+ HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection(Ye Maosheng, cite100, 2020)
+ Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds([Lin Dahua](https://www.aminer.cn/profile/lin-dahua/53f42cf2dabfaedce54bedd8), cite0, 2020)

### 2. pillars:

pillar可以看作是一种特殊的voxel, 它的垂直方向是无限大。先通过PointNet将点云聚合成pillar特征，然后分散后构成2D BEV图像，之后对2D图像进行特征提取。

+ 首次提出pillar概念的文章：PointPillars: Fast Encoders for Object Detection from Point Clouds(Alex H Lang, cite1600, 2019)
+ 基于PointPillars的演进方法有：
  + Pillar-Based Object Detection for Autonomous Driving([Tom Funkhouser](https://www.aminer.cn/profile/tom-funkhouser/53f438fddabfaeecd6977b07), cite100, 2020)
  + Embracing Single Stride 3D Object Detector with Sparse Transformer(Zhaoxiang Zhang, cite30, 2022)

### 3. BEV

BEV特征地图(BEV feature map):  BEV特征图是种稠密的2D表示，每个像素对应一个特定的区域，并对该区域内的点信息进行编码。可以通过将voxel或pillar的3D特征映射到BEV或者通过统计像素区域内的点的信息来产生BEV特征图。基于BEV的3D检测方法有：

+ PIXOR: Real-time 3D Object Detection from Point Clouds([R Urtasun](https://scholar.google.com/citations?user=jyxO2akAAAAJ&hl=zh-CN&oi=sra), cite700, 2018)
+ HDNET: Exploiting HD Maps for 3D Object Detection([R Urtasun](https://scholar.google.com/citations?user=jyxO2akAAAAJ&hl=zh-CN&oi=sra), cite200, 2018)
+ RAD: Realtime and Accurate 3D Object Detection on Embedded Systems(Robert Laganiere, cite2, 2021)
+ Multi-View 3D Object Detection Network for Autonomous Driving(Xiaozhi Chen, cite1800, 2016)
+ BirdNet: A 3D Object Detection Framework from LiDAR Information(Barrera Alejandro, cite180, 2020)
+ YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud(Waleed Ali, cite80, 2018)
+ Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds(Horst-Michael Gross, cite200, 2018)
+ Vehicle Detection from 3D Lidar Using Fully Convolutional Network(Bo Li, cite500, 2016)

以上提到的基于grid的数据形式的3D检测方法分类如下：

![image.png](https://s.readpaper.com/T/1rTAq63nHFo)



### 基于grid的lidar 3D检测算法总结

1. voxel携带更多深度信息，但是对应的计算和存储资源消耗更多；
2. 基于BEV特征图的方法速度快，但是检测精度比基于voxel的低；
3. 基于pillar的精度和效率介于BEV和voxel之间；

基于grid方法面临的同一个问题是：如何选取一个合适的grid size？

+ grid size较小时，携带的信息粒度更精细，但是存储和计算量较大；
+ grid size较大时，存储和计算效率高，但是检测精度较低；



## 结合point和voxel的算法

point和voxel结合的检测方法的基本框架如下图所示：
![image.png](https://s.readpaper.com/T/1rU2fsUE5fE)
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
![image.png](https://s.readpaper.com/T/1rUcgNtHRT7)

### point-voxel结合方法的总结

此类方法的检测精度比纯voxel的高，但是计算成本和难度增加，point-voxel的关联耗时且不准确，同时点云特征和3D proposal的配合也难度较大。



## 基于range的3D检测

range图像是一种稠密的2D表示，每个像素携带了距离信息，而不是RBG信息。基于range图像的3D检测方法主要从两个方面来完成3D目标检测任务：

1. 设计专用于range图像的模型和算子；
2. 选择合适的view；

基于range图像的3D检测框架如下图所示：
![image.png](https://s.readpaper.com/T/1rWAhtai3oO)

### 基于range的模型

因为range和普通camera的图像都是2D的，所以常将2D检测模型移植到range图像上。

+ 开创性的工作：LaserNet，使用DLA-Net(deep layer aggregation network)来获得多尺度的特征，并从range图像中检测3D目标， 论文题目：LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving(Gregory P. Meyer, cite200, 2019)
+ 演进工作：
  + RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation（Zhidong Liang, cite40, 2020）
  + RSN: Range Sparse Net for Efficient, Accurate LiDAR 3D Object Detection([Drago Anguelov](), cite50, 2021)
  + Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection([Drago Anguelov](), cite20, 2020)
  + RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union(Zhidong Liang, cite20, 2021)
  + RangeDet: In Defense of Range View for LiDAR-Based 3D Object Detection(Zhaoxiang Zhang, cite60, 2021)

### 基于range的算子

由于range图像和RGB图像中每个像素携带的信息不同，所以使用的算子也不同。一些方法通过改善算子来提高特征提取的效率和精度，代表性方法如下：

+ Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection([Anguelov Dragomir](), cite30, 2020)
+ To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels([Anguelov Dragomir](), cite20, 2021)

### 基于range view的方法

range图像是由点云进行球状映射得到的，基于range视角的检测会存在遮挡和比例尺变化的情况。为解决此问题，很多文章尝试从其他view来进行3D检测，代表性方法有：

+ It's All Around You: Range-Guided Cylindrical Network for 3D Object Detection(Dan Raviv, cite20, 2021)

以上提到的各方法的分类如下：
![image.png](https://s.readpaper.com/T/1rWWZLYaV1c)

###  **关于range 3D检测方法的总结**

基于range的检测方法效率高，但是容易受遮挡和尺度变化影响。
因此，当前流行的基于range的范式是从range中进行特征提取，从BEV进行3D检测。



## 基于anchor的lidar 3D检测

1. 3D目标比较小；
2. 点云稀疏，难以检测和3D目标尺寸估计；

anchor是预定义尺寸的长方体，可被置于3D空间的任意位置。
假设ground truth为$[x^g, y^g, z^g, l^g, w^g, h^g, \theta^g], cls^g$;
anchor为$[x^a, y^a, z^a, l^a, w^a, h^a, \theta^a]$;
由anchor预测得到3D检测框$[x, y, z, l, w, h, \theta]$.
基于anchor的3D检测基本流程如下图所示：
![image.png](https://s.readpaper.com/T/1rZcJOgr6sT)

我们从两个方面来介绍基于anchor的解决上述问题的技术：

1. anchor的配置：基于anchor的3D检测方法基本都基于BEV,3Danchor放置于BEV特征图中每个grid中，并且每个类别的anchor都有固定的尺寸；

   **问题：每个类别固定尺寸的anchor是否合理？**

2. 损失函数：$L_{det} = L_{cls} + L_{reg} + L_{\theta}$, 其中：

   + $L_{cls}$是positive anchor和negative anchor的分类回归损失；
   + $L_{reg}$是3D尺寸和位置的损失函数；
   + $L_{\theta}$是heading的损失函数；

   voxelNet是第一个将3D IOU和损失函数结合起来的文章，之后还有基于focal loss的方法，基于SmoothL1的方法，基于corner loss的方法等。

### 基于anchor检测方法的挑战

1. 对于小目标，使用anchor难度大；

2. grid大时，anchor一般也大， 如果目标小则iou小；

3. 实际应用中，需要的anchor数量太多；

   

## anchor-free的3D目标检测

anchor-free的检测方法不需要设计复杂的anchor机制，可以灵活应用于BEV, point view, range view等多种视图。基于anchor-free的3D检测基本流程如下图所示：
![image.png](https://s.readpaper.com/T/1rZcrTCCh5U)

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

### 关于anchor-free方法的总结

anchor-free方法灵活简单，各种anchor-free方法中，最具潜力的方法是center-based方法：Center-based 3D Object Detection and Tracking，其在小目标领域表现良好，并且超过anchor-based基线。
anchor-free方法的难点在于，需要设计一个准确的过滤机制，将bad positive样本过滤掉。在这个问题上，anchor-based方法只需要计算iou，设置iou阈值即可。



## 通过辅助手段来改善3D检测的方法

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
   + Point Density-Aware Voxels for LiDAR 3D Object Detection
   + GLENet: Boosting 3D Object Detectors with Generative Label Uncertainty Estimation
4. 目标局部预测：获取目标的局部信息有利于3D检测，相关的文章有：
   + Object as Hotspots：
   + From Points to Parts

### 关于3D检测辅助手段的总结

还有很多辅助手段来提高检测精度，比如场景流估计(scene flow estimation)等。



## 基于LiDAR的3D检测算法展望

基于LiDAR的3D检测算法近年来取得极大进展代表性进展如下：

+ 基于pillar的算法运算速度非常快，可达60Hz以上；
+ 基于voxel和point结合的方法精度很高，比如PV-RCNN系列在KITTI上对car easy样本的AP达到90.25%；
+ SE-SSD对KITTI的car easy样本的AP达到91.49%；
+ 基于点云补全的方法GLENet在KITTI上对car easy的AP达到91.67%；
+ 基于Transformer的方法DSVT在Waymo上对vehicle L1的样本的mAP达到80.3；

这些方法都有各自的优点，总的来说，这些算法的演进方向符合如下节奏：

1. 2016-2019年间，学者们在研究如何高效提取点云信息，发展出各种框架，如VoxelNet类、PointPillars类、PointNet类的算法；
2. 2020-2021年间，学者们试图将RCNN、voxel、point的优点结合，比如PV-RCNN、M3DeTR等方法；
3. 2021-2023年间，进入百花齐放阶段：
   + 有尝试将Transformer应用于LiDAR 3D检测的，比如CT3D/VoTr/DSVT等；
   + 有尝试对输入点云做增强的方法，比如PDA/BtcDet/GLENet等；
   + 有将目标检测转换为序列解码的方法，比如Point2Seq;
   + 有尝试多模融合的方法，比如3D-Dual Fusion；

对于未来的技术演进路线，可能会是当前探索到的多种有效方法的结合，比如将点云增强应用于Transformer检测结构、比如多模融合技术与多种数据形式的技术结合；也可能某一单点技术会有更突破性的进展，比如基于Transformer的检测方法。

总的来说，经过最近几年的技术积累和技术探索，接下来将会涌现出更具竞争力、更具统治力的检测算法。





















