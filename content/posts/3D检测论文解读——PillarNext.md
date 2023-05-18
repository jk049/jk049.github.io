---
title: "3D检测论文解读——PillarNext"
date: 2023-05-17T18:25:54+08:00
description: "想要重塑基于pillar的3D检测基线框架的一篇文章"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 213030.png"
tags: ["next基线框架","3D","Pillar","point"]
theme: "light"
---

# 一句话总结

这篇文章想要颠覆已有点云3D检测框架，提出了新的基于pillar形式的点云3D检测框架，而且作者指明后续演进方向是将2D检测的优秀成果嫁接到基于pillar的3D检测领域。

但是呢，这篇文章的作者是杨晓东，先在Nvidia工作，后加入了轻舟智航。他在3D检测领域可以说是名不见经传的人，一上来就这样的口气难免让人怀疑在说大话。

总之，这篇文章的作者做到了大胆假设，我们读者要小心求证。如果自行验证结果与论文观点一致，那么这篇文章就真的可以称为宝藏了，且让我们拭目以待。

# 摘要

**背景**：为了解决点云的稀疏性和非规则分布问题，基于lidar的3D检测方法大都设计局部点云聚合机制来进行细粒度的几何建模。

**PillarNext的方法**：该文章从计算资源的角度重新审视了局部点云聚合机制。 作者发现最简单的pillar架构在精度和延时方面的表现出奇地好。 此外，作者发现只要对2D检测领域中的成熟组件做一些小的适配，比如提高感受野等技术，就可以大大提高检测精度。

**实验结果**：在WOD和NuScene上SOTA。该实验结果与直觉相反，人们直觉认为详细的几何信息建模对3D检测至关重要。然而本实验结果显示：对于3D检测来说，也许并不需要详细的几何信息。

代码仓：https://github.com/qcraftai/pillarnext

# 基本介绍

**传统3D检测算法**：3D检测对自动驾驶很重要，包括感知、预测、规划等领域。 因为点云分布的非规则性，之前的方法都用各种方法非规则分布点云转换为规则分布的数据形式，比如pillar系列、voxel系列、range系列等。 然而，这种数据形式转换过程会不可避免导致信息丢失，进而对检测结果有影响。 最近PV-RCNN++采用voxel/point结合的设计，从而进行细粒度的几何信息建模。 

【注】这已经不是最近了吧？这个信息滞后有点多哦。。。

该文的设计团队将上述基于pillar、voxel、range的设计统称为局部点云聚合方法，因为这些方法采用各种技术对点云在特定邻域内进行特征聚合。 该团队还认为：当前的点云3D检测算法的演进都是提出各种算子，而对检测框架的演进几乎停滞。大多数算法还是基于SECOND结构或PointPillars结构，在这方面缺乏新的设计。

**2D检测领域的进展**：与3D检测相比，2D检测领域取得极大进展，这些进展归因于架构和训练技术的演进。2D检测的优秀BackBone有ResNet、SwinTransformers，高效的neck有BiFPN、YOLOF，训练方法有bag of freebies。 

**PillarNext的设计思路**：鉴于以上阐述，PillarNext团队认为应该重新审视点云3D检测应该聚焦于哪个点。具体来说，要重新审视局部点云特征聚合和整体架构两个方面：

- 对局部特征聚合模块的分析：首先，团队从计算资源的角度对局部点云特征聚合进行了比较。细粒度的特征聚合往往比粗粒度的聚合消耗更多计算资源。举例来说：基于voxel的聚合模块采用3D卷积的方式，比2D卷积运行更慢、需要的网络参数更多。这引出一个问题：我们应该如何分配资源？下图是各方法在计算资源与检测效果方面的比较：![image-20230518111906451](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518111906451.png)由图中可知：基于pillar的方法比基于voxel和多角度融合的方法的表现都好。这一现象相当于对3D检测框架中细粒度的局部聚合的必要性提出质疑。 该团队认为：他们的这一发现与PointNext、A Closer Look at Local Aggregation Operators in Point Cloud Analysis的分析一致，这两篇文章认为各种局部特征聚合的效果都差不多，整体框架的影响力更大一些。

- 对整体架构的分析：该团队的设计理念是基于优秀的2D框架进行尽量少的适配，而不是提出新的设计。举例来说，适当提高感受野进而显著提高检测效果。与PillarNet、MVX-Net的多尺度特征融合技术不同，该团队论证了只要在最终阶段的感受野够大，则使用单一尺度的特征也可以取得很好的检测效果。这一发现表明：3D检测领域可以使用2D检测领域的成熟技术。

  【注】这两个观点是否能站住脚，还需要去详细分析论证。

**PillarNext设计结果**：基于上述的分析，该团队提出PillarNext框架，在WOD和NuScene上SOTA。 基于2D检测技术的发展，该团队发现增加感受野大小对3D检测非常重要。进而对2D检测框架进行了少量适配，从而在3D检测领域取得很不错的检测效果。

# 相关工作

**之前的点云3D检测方法**：

- PointRCNN: PointRCNN使用PointNet机制产生proposal，然后使用ROI Pooling对proposal进行refine。但是这种方法的点云特征聚合消耗的计算量非常大，步适用于大规模点云场景；
- VoxelNet: 将点云特征聚合到每个voxel内，然后用稠密3D卷积进行上下文建模；
- SECOND：引入3D稀疏卷积对VoxelNet进行优化；
- PointPillars: 将点云划分到pillar形式，然后使用2D卷积；
- LaserNet:基于range图像使用2D卷积；
- 结合point和grid的方法：PV-RCNN等。从直觉来看，将点云转换为其他形式会导致信息丢失，因此point、voxel结合的方法认为能提高检测效果。但是该团队的观点是：细粒度的局部特征聚合的作用被高估了，因而聚焦于整体架构和训练方式。

**特征融合**：多尺度特征融合技术始于FPN，之后被广泛用于2D检测领域。FPN按自顶向下的顺序将高维度的语义特征与低维度的空间特征结合，从而提高检测精度。 PANet进一步指出自底向上的多尺度特征融合也是很重要的。 BiFPN不同尺度特征的贡献是不一样的，然后采用学习的方式调节各尺度特征的权重。

**感受野**：ASPP(Atrous Spatial Pyramid Pooling)技术对多个感受野的特征进行采样，TridentNet使用三层不同dilation factor的卷积层来时感受野匹配不同尺度的目标。YOLOF使用dilated residual模块来增加感受野。 这些特征融合和感受野技术已经在2D检测领域广泛应用，但是在3D检测领域的讨论还很少，大多数框架还是基于VoxelNet。该团队就是想将2D检测领域的最新技术应用到3D领域中。

**Model Scaling**:2D检测领域已经验证深度、宽度、分辨率对检测精度的影响很大。EfficientNet提出一种缩放规则，之后被应用于EfficientDet和Fast and Accurate Model Scaling。在3D检测领域，只有Deep Supervised Hashing with Triplet Lables这一篇文章分析了model scaling的影响。但是该文章只针对SECOND进行了分析，PillarNext对此进行了系统的比较和分析。

# PillarNext设计细节

该团队基于grid方法构建PillarNext，因为grid方法效率更高，同时整体结构也与2D检测框架更相似。 通常，grid方法结构如下：

- grid encoder：将原始点云转换为规则化的结构化的特征图；

- BackBone: 通用特征提取；

- neck：用于多尺度特征融合；

- head：针对具体任务输出结果；

  

PillarNext整体架构如下图所示：![image-20230518112001745](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112001745.png)

接下来详细阐述各模块的设计细节。

## Grid Encoder

Grid Encoder的作用是将离散化无规则分布的点云转换为规则化分布的grid形式，然后将grid内的点云进行转换和聚合，生成初步的特征。 grid形式有voxel、pillar和Multi-View Fusion三种。该团队基于Pillar-based Object Detection的grid机制，将pillar和cylindrical的形式结合。

**编码效果对比**：结论是Pillar形式的框架不比voxel的表征形式差。

为验证各种grid encoder的效果，该团队以稀疏ResNet18为BackBone，变化网络规模，得到tiny/small/base/large四种规模的网络。实验数据如下表所示：![image-20230518112025149](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112025149.png)

![image-20230518112035047](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112035047.png)

由上图可知，对于BEV的AP，pillar编码的与voxel编码的精度差不多，但是速度远远快于voxel形式的；但是3D指标下Pillar精度全面低于voxel精度。该设计团队将训练epoch提高到36，增加IoU回归损失和IoU Score Branch，得到一系列的Pillar+/Voxel+模型。这些后处理优化的模型的检测结果表明，pillar编码方式的检测精度不劣于voxel编码方式。

该团队猜想：由于没有高度信息，所以pillar模型需要更长的训练时间才能收敛。同时，该实验证明了细粒度的局部几何模型建模不是必须的。此外，该实验还证明，将计算资源分配给高度信息上没什么用，因此该团队将所有计算资源分配在BEV平面上。

**编码训练对比**：结论是pillar形式的框架比voxel框架需要更多epoch才能收敛。

实验数据如下图所示：![image-20230518112058619](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112058619.png)

由此图可知：对于BEV检测指标，Pillar和voxel编码方式的训练收敛速度差不多，但是对于3D检测指标，pillar编码模型的收敛速度要明显慢于voxel编码的模型。所以之前研究的pillar模型精度差于voxel精度的主要原因是收敛速度，而不是细粒度的特征聚合。将pilllar模型训练到充分收敛后，其检测精度应该于voxel模型差不多。

【注】这个结论太颠覆了，可不要随便说。做更多实验再回头看吧，MARK！

## BackBone

BackBone基于grid encoder的初步特征进行进一步的特征提取。考虑到ResNet18在2D检测领域的广泛应用，该团队采用ResNet18作为BackBone。

**关于分辨率的研究**：直觉来讲，grid的尺寸越小，保留的细粒度的信息越多，但是对应的计算消耗也越大。下采样能有效降低计算消耗，但是也会降低检测精度。该团队设计了不同尺寸的grid和特征，通过改变BackBone和Head中的grid尺寸及特征采样倍率来进行实验。实验结果如下表所示：![image-20230518112142607](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112142607.png)

由该实验得知：

- 输出特征分辨率固定为0.3时，改变grid的尺度对大目标的检测精度几乎没影响，但是会明显影响小目标的检测精度——grid尺寸越小，小目标的检测精度越高；

- 改变输出特征的下采用分辨率，会影响所有目标的检测精度：然而只要在head里加一个上采样层，就能大大提高检测精度——这似乎说明：细粒度的特征信息已经编码到下采样特征中，只需要简单的上采样就能恢复那些细粒度特征；

  

## Neck

Neck的作用是对BackBone的特征进行聚合，进而增加感受野、融合多尺度的上下文信息，进而应对不同尺寸的目标。PillarNet、MVX-Net等架构都使用了多尺度特征融合的设计，对不同阶段的特征图进行上采样到同一分辨率后级联起来。

但是有没有更高效的特征聚合方式呢？图像检测领域详细研究过如何设计高效、有效的Neck，但是3D检测领域还未对neck进行详细研究。

**Neck设计实验**：该团队将2D领域的先进Neck设计适配到3D检测领域，比如BiFPN、ASPP等。简单地将BiFPN和FPN技术集成到PillarNet之后，对车的检测精度比原始的PillarNet提高了2.38个百分点。数据如下表所示： ![image-20230518112202099](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112202099.png)

该设计团队说：多尺度特征聚合的目的是提高对不同大小的目标的检测鲁棒性，但是基于BEV空间的3D检测不存在此问题。

【问】why?

基于上述的观点，该团队开始思考：3D检测到底需不需要多尺度特征聚合？为此，该团队做了3个单尺度neck的实验：

- 基线：使用残差模块(residual block)作为neck，没有上采样和下采样，感受野有限，精度较差；
- 增加dilated block：YOLOF中说：对于2D检测，当感受野和目标尺寸差不多大小时，检测精度较高。因此该团队也使用YOLOF中的dilated模块来提高感受野大小，从而提高了检测精度；
- 增加ASPP模块：将DeepLab的ASPP模块集成到neck中，又提高了2.45个百分点；

**结论**：上面3个实验表明：多尺度特征聚合不是必须的，单尺度的neck就可以表现得更好。因此给3D检测领域的neck设计打开一个思路：探索更多2D检测领域的优秀neck设计，将其移植到3D检测的neck上，说不定会有更大的提升。

## Head

SECOND和PointPillars的框架的head基于anchor，将feature map与anchor进行映射；CenterPoint使用目标质心代替每个目标，通过质心热力图回归检测框。PillarNext采用CenterPoint的head机制，然后进行了特征上采样、multi-grouping和IoU Branch等优化，最终显著提高了检测指标。

# 实现细节及实验结果

**数据集**：在WOD和NuScene上训练和测试。

- WOD：包含1000个序列，其中点云由5个lidar以10hz的帧率采集。使用AP和APH作为评价指标，车、人、自行车的IOU分别为0.7、0.5、0.5。

- NuScene：包含1000个序列，每个序列时长20s，由32线lidar按20hz的帧率采集点云。评价指标采用10类目标的mAP和NDS(NuScene Detection Score)。

  

**实现细节**：基于PyTorch框架，优化器采用AdamW。WOD的水平检测范围是[-76.8, 76.8]，垂直检测范围是[-2, 4]，Pillar尺寸是0.075m，训练epoch为12，连续3帧输入？车、人、自行车的NMS阈值为0.7， 0.2， 0.25。NuScene的水平检测范围是[-50.4, 50.4], 垂直检测范围是[-5, 3]，所有类别的NMS阈值为0.2，其他参数与WOD的一样。推理卡为单张Titan RTX。weight decay比例为0.01，学习率为0.001，batch size是16。数据增强方式包括反转、旋转-90~90度、缩放0.9-1.1，随机加噪。具体信息如下表：

![image-20230518112226607](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112226607.png)

**各组件的实验效果**：具体信息如下图所示：![image-20230518112248753](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112248753.png)

由实验结果可知，适当的grid尺寸、增强的neck、增强的Head、合适的训练都能有效提高检测精度。

**开源数据集上的检测指标对比**：具体数据如下表所示：![image-20230518112307487](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112307487.png)

![image-20230518112317077](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112317077.png)

![image-20230518112326484](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112326484.png)

![image-20230518112337559](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230518112337559.png)

由该表可知：PillarNext比很多二阶段算法的检测精度还高，甚至比带有时序信息和细粒度几何结构建模方法的检测精度高。

【注】从给出的表格看，数据还可以。但是U1S1，表格里的对照方法大多是老旧方法，没什么竞争力。笔者不禁怀疑：作为2023年5月的文章，该团队是不是故意不与新模型对照，刻意避开那些有竞争力的模型？ 

# 个人分析与总结

**作者信息**：QCraft的杨晓东团队发表于2023年5月。杨晓东之前在Nvidia任高级研究科学家，后加入轻舟智航。他之前几乎没涉猎点云3D检测，这篇文章可以说是初出茅庐就要颠覆传统。

再回头说轻舟智航：这家公司成立之初的目标是商业落地L4自动驾驶，从Waymo、Tesla、Facebook、Apple、Nivida等公司挖了很多人。近几年L4迟迟看不到落地希望，这家公司快速转身，保持L4目标之外迅速开始一系列的L2辅助驾驶落地动作。从技术角度，点云3D检测领域几乎没看到该公司发表什么文章。事实上，工业界里频繁提出3D检测算法并发表文章的公司几乎只有TuSimple一家。

**创新点**：提出一种新的基于pillar的3D检测框架，做了一系列实验验证各模块的作用，论证主流框架里流行的多尺度特征聚合没什么用，基于单尺度特征，用2D检测的一些优秀Neck就能有足够的感受野，进而提高检测指标。

结果方面，WOD vehicle L1的AP能达到83以上，还不错。

**大话演进方向**：上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持批判态度阅读。

分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。

但是后续我会进行独立实验及相关的研究跟踪，预知后续发展请保持关注。

话不多说，上菜：

- 小心求证：这篇文章的观点很具有颠覆性，抛开技术和指标不谈，笔者认为最终要的是先验证文章中的观点是否能站住脚，之后才能谈演进和优化的事；
- 2D检测领域的其他优秀技术“嫁接”：工程上应该称为适配，但是笔者觉得“嫁接”这个词很形象，姑且用这个词。目前PillarNext使用了2D neck的一些技术，比如ASPP，BiFPN，BackBone使用ResNet18，其他都是些小东西。如果论文中的观点被证明是有效的，那么真的可以按照作者说的：以PillarNext为框架，尝试更多的2D检测技术；
- 全稀疏概念：最近全稀疏3D检测的概念比较火，另一个“next”字辈模型就是基于全稀疏概念设计的，pillarNext是否也可以结合全稀疏框架？
- LC融合：最近基于LC融合的3D检测方法的效果慢慢反超基于单lidar的指标了，很多榜首模型都用了各种融合技术，比如LoGoNet的局部融合、VirConv的图像稠密伪点云去冗余技术。如果PillarNext被证明有效，那么基于此框架进行LC融合，是否能进一步提高3D检测指标？