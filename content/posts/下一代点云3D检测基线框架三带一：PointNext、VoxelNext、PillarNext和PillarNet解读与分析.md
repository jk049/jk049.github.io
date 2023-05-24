---
title: "下一代点云3D检测基线框架三带一：PointNext、VoxelNext、PillarNext和PillarNet解读与分析"
date: 2023-05-24T21:43:11+08:00
description: "下一代点云3D检测基线框架"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-18 221345.png"
tags: ["next","3D","point"]
theme: "light"
---

# 一句话总结

本文对4个典型的"下一代"点云3D检测基线框架(PointNext/VoxelNext/PillarNext/PillarNet)进行了全面的分析与对比，从而使读者对点云3D检测算法的演进方向有自己的判断，进而以这种演进趋势来重新审视之前提出的经典算法，最终对点云3D检测算法的发展方向有更深刻的认识和判读。

# 三带一框架基本介绍

由于自动驾驶对高精度感知的需求，学术界和工业界对点云3D检测的投入很大，各种检测算法也层出不穷。点云3D检测算法里程碑如下图所示：

![点云3D检测算法里程碑](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/基于Lidar的3D检测方法总结.svg)

从时间角度看，2017、2018年可以称为点云3D检测元年，主要的3D检测框架PointNet、VoxelNet、PointPillars都是这段时间提出的。基于这三种框架的算法或独自优化，演进出SECOND、VoxelRCNN、CT3D、VoTr等算法；或交叉优化，演进出PV-RCNN系列方法。

这一演进趋势持续到2022年，再之后的方法很少对传统的框架做改进，而是提出五花八门的方法，比如点云补全方法SPG、GLENet等方法, 基于序列的检测方法MPPNet，用图像生成虚拟点云来增强原始点云的方法SFD、VirConv，将点云目标检测转换为序列检测问题的Point2Seq，全稀疏检测方法SST、FSD等。这一新趋势仿佛告诉人们：传统框架的潜力已经挖掘完了，在检测精度上碰到了天花板。

2022年、23年可以称为点云3D检测的春秋战国时期，各种方法百家争鸣。在这诸子百家中，有一个流派与众不同：别的流派都是在经典框架下引入各种新方法来寻求指标上的提高，但是有一个流派致力于对基线框架做改进，这就是"next"流派。"next"流派针对PointPillar、PointNet、SECOND这三种点云3D检测基线算法做架构上的优化，提出PillarNext、PointNext、VoxelNext等框架，试图以这几种框架作为新的基线框架，让后续的点云3D检测算法都以这几种"next"框架为基线框架进行优化演进。

不论这几种"next"框架是否有效，这种思路值得我们留意，毕竟一旦证明某一"next"框架有效，将对对应的方法产生全面的颠覆性的影响。本节接下来对PointNext/VoxelNext/PillarNext/PillarNet这四个框架进行基本介绍，使读者对各框架的整体结构及优化思路有基本的了解。

笔者之前对这4篇文章进行了详细的解读，具体内容见如下链接：

[3D检测论文解读——PointNext](https://jk049jk.com/posts/3d%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BBpointnext/)

[3D检测论文解读——VoxelNext](https://jk049jk.com/posts/3d%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BBvoxelnext/)

[3D检测论文解读——PillarNet](https://jk049jk.com/posts/3d%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BBpillarnet/)

[3D检测论文解读——PillarNext](https://jk049jk.com/posts/3d%E6%A3%80%E6%B5%8B%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BBpillarnext/)

本文接下来只对这4个框架进行简单介绍。

## PointNext基本介绍

PointNext是阿卜杜拉国王科技大学的Bernard Ghanem和钱国成团队提出的，Bernard大佬之前提出过点云3D检测算法PointRGCN。

这篇文章的主要观点是认为PointNet++的潜力还没被完全挖掘出来，因此从两方面进行优化：

- 应用先进的训练策略，比如AdamW优化器、交叉熵损失函数、数据增强技术等；
- 改进PointNet++框架，提出Inverted Residual MLP和感受野扩张技术；

该框架在语义分割和目标分类数据集上进行了验证，没有在3D检测数据集上验证。

**针对训练技术的分析**：受益于机器学习理论的发展，现在的神经网络可以用更好的优化器，比如AdamW；还有更高级的损失函数，比如CrossEntropy with label smoothing。 该团队对各优化技术进行了控制变量实验，实验数据如下所示：

![image-20230519170114194](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170114194.png)

实验结论是：CrossEntropy with label smoothing， AdamW, Cosine Decay在大部分场景下都表现优秀。

**针对PointNet++架构的优化**：PointNext对框架的主要改动包含两个方面：感受野扩张(receptive field scaling)和模型扩张(model scaling)。

+ 感受野扩张：具体方式是通过一系列实验设置最优的查询半径，此外该团队对相对位置进行归一化处理，降低了优化难度。 
+ model scaling: 该团队发现无论是增加更多SA模块还是增加更多通道，都会导致梯度消失和过拟合，不能提高精度。进一步分析之后，该团队提出Inverted Residual MLP模块，级联在SA模块之后，被证明能有效实现model scaling。整体结构如下图所示：

![image-20230523093130912](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230523093130912.png)

总的来说，这篇文章应该称为PointNet plus pro，而不是“next”。因为是基于PointNet++的演进，并没有完全替换或大规模改动PointNet++的结构。个人认为，该文章更有价值的工作是对PointNet++各种组件及参数进行了详细的分析，不论对框架的改动是否有效果，这些验证分析工作值得我们思考。

Anyway, 这篇文章为我们打开了PointNet系列的演进思路，值得我们进一步尝试。

## VoxelNext基本介绍

这篇文章是港中文的贾佳亚团队、香港大学、旷世科技张翔雨团队合作的3D检测算法，主要创新点是：将经典的SECOND结构改为极简架构，以全稀疏的方式直接从voxel特征检测目标。实验指标也不错，在NuScene上的检测效果优于CenterPoint，劣于PillarNet，但是速度很快。

值得注意的是：贾佳亚大佬之前提出很多3D检测算法，比如STD、IPOD等，这些方法都有一个特点：结构复杂、计算量大、提升较小。可以说：之前贾大佬提出的各种点云3D检测方法都属于吃力不讨好类型，不适合工业化落地。现在这篇极简架构的VoxelNext倒是一改大佬以往风格，提出了兼具效果和易落地的算法。具体效果如何，让我们拭目以待吧。

该团队的设计思路是：对SECOND结构进行最小修改，实现voxel形式的高效、准确的3D检测。框架主要包含３个改动点：BackBone，稀疏检测头和跟踪模块。下面分别进行详细介绍：

+ **BackBone优化**：对BackBone的优化主要有3点：

  + 增加下采样层：基于稀疏的voxel特征直接进行精确的3D检测要求特征必须表征能力强且感受野够大。虽然3D稀疏卷积特征用于3D目标检测已经持续很多年了，但是最近的研究表明其特征信息不够充分，需要用Focal Sparse Convolution、Transformer等机制进行进一步增强。针对此问题，该团队的解决办法是增加一个下采样层。为对特征进行增强，我们对原结构增加两个下采样层，对应的步长是16和32。这个小改动极大地增加了感受野，效果示意图如下所示：![image-20230522094911441](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094911441.png)
  + 稀疏2D特征：典型3D检测框架将3D特征转换为2D稠密特征，但是该团队发现2D稀疏特征就足以进行3D检测。因此该团队将voxel特征投影到平面上，相同位置的特征相加。
  + voxel剪枝：3D点云包含大量背景点，对目标检测没用。该团队用下采样层不断裁剪没用的voxel，按照VS-Net中SPS-Conv的机制，有效控制了voxel数量。

+ **检测头优化**：直接基于3D稀疏特征进行检测，训练时，将voxel assign到最近的真值框，损失函数使用Focal Loss, 推理时使用max-pooling来代替NMS。与Submanifold稀疏卷积类似，max-pooling只在非空地方执行，从而节省了head的计算量。

  此外，按照CenterPoint的原则，对平面位置、高度、朝向角进行回归。对于NuScene数据集或跟踪任务，还回归速度信息。这些预测结果的损失函数基于L1损失函数。对于Waymo数据集，对IOU的预测通过kernel size为3的submanifold卷积层。该方式比全连接层精度高，时间消耗差不多。

+ **跟踪模块优化**：跟踪模块是基于CenterPoint的扩展，基于速度对目标位置进行预测，基于L2距离计算关联关系。

整体结构如下图所示：

![image-20230523103850333](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230523103850333.png)

VoxelNext主要改动点包含两个方面：全稀疏框架和增加感受野。其中，全稀疏框架主要改动在于2D特征由稠密特征改为稀疏特征，以及针对2D稀疏特征的一系列适配；增加感受野主要通过在3D BackBone中增加下采样层来实现。对于全稀疏框架来说，最近很热的技术还有FSD、SST等，或许VoxelNext可以考虑这些更全面更系统的全稀疏技术。对于感受野扩张技术，下采样层的设计很颠覆常识，我们读者可以进一步对其验证，同时这也是3D检测领域共同面对的问题，我们也可以看看别人是怎么解决这个问题的。

总的来说，之前Voxel框架都基于SECOND结构，先3D稀疏卷积，然后转成2D稠密特征后进行2D卷积，再接proposal、head那一套流程。VoxelNext的全稀疏voxel框架使用2D稀疏特征进行检测，改进了NMS机制，从而简化了结构、节省了时间。我们可能需要对该框架进行进一步的优化和打磨，运气好的话，也许真的可以优化出实用、好用的下一代voxel形式的3D检测基线框架。

## PillarNet基本介绍

PillarNet是上交马超团队于2022年5月发表的一篇基于pillar的点云3D检测算法，马超团队近年来一直在零星发表3D检测相关文章，比较知名的是LC融合3D检测方法PointAugmenting。

该文章主要创新点是将pillar特征编码由PointNet的MLP方法改为2D稀疏卷积方法，对Neck引入特征聚合机制，对head进行了小改动。

该团队认为Pillar方法比SECOND方法检测精度差的主要原因在于稀疏特征编码学习到的空间特征不够有效、neck模块的空间-语义特征融合不够有效。此外，PointPillars输出的特征图尺寸与点云范围、pillar尺寸有关系，不适用于大维度点云。 于是该团队对PointPillars进行该进，其中主要改进点为：

- **2D稀疏卷积**：在2D平面也使用稀疏卷积提取BEV特征。特征编码模块采用4层2D稀疏卷积，下采样倍数分别为1、2、4、8。这种设计可以使用图像领域广泛应用的VGGNet/ResNet等结构，同时下采样设计可以提供不同pillar尺寸的特征。

- **neck语义特征抽象**：在低分辨率特征图上进一步进行语义特征抽象；FPN结构中的neck模块的作用是：将高维度的语义特征与低维度的空间特征进行融合，进而将融合特征输入給head。 PillarNet基于SECOND的Neck提出两种neck设计，如 下图所示：

  ![image-20230522153600230](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522153600230.png)

PillarNet整体结构如下图所示：

![image-20230523113102075](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230523113102075.png)

马超团队在3D检测领域投入不大，会零星发表几篇相关文章，而且大部分文章都是基于LC融合的方法。相对有名的文章是PointAugmenting和LIFT。从这几篇文章我们可以看出马超团队的风格：通过对已有的简单框架进行简单优化，来尽可能提高检测结果。这种风格正好适合工业场景的落地，我们不妨对此文章保留一点期待。

有意思的是，PillarNet发表一年之后，QCraft的杨晓东团队发表了PillarNext。仔细分析后我们可以发现，二者有很多相似性，让我且往下看：

## PillarNext基本介绍

该团队的设计理念是基于优秀的2D框架进行尽量少的适配，使之能够解决3D检测问题。该团队基于grid方法构建PillarNext，因为grid方法效率更高，同时整体结构也与2D检测框架更相似。 该团队从以下4个方面对检测框架进行分析:

+ **编码形式分析**：人们直觉认为详细的几何信息建模对3D检测至关重要。然而该团队的实验结果显示：对于3D检测来说，也许并不需要详细的几何信息。该团队通过一系列实验证明pillar的编码方式不影响3D检测精度。

  由于没有高度信息，所以pillar模型需要更长的训练时间才能收敛。同时，还证明了细粒度的局部几何模型建模不是必须的。此外，将计算资源分配给高度信息上没什么用，因此该团队将所有计算资源分配在BEV平面上。

+ **BackBone分析**：BackBone基于grid encoder的初步特征进行进一步的特征提取。考虑到ResNet18在2D检测领域的广泛应用，该团队采用ResNet18作为BackBone。直觉来讲，grid的尺寸越小，保留的细粒度的信息越多，但是对应的计算消耗也越大。下采样能有效降低计算消耗，但是也会降低检测精度。此外，该团队还进行了grid尺寸的实验分析，对应结论为：

  - 输出特征分辨率固定时，改变grid的尺度对大目标的检测精度几乎没影响，但是会明显影响小目标的检测精度——grid尺寸越小，小目标的检测精度越高；
  - 改变输出特征的下采用分辨率，会影响所有目标的检测精度：然而只要在head里加一个上采样层，就能大大提高检测精度。这似乎说明：细粒度的特征信息已经编码到下采样特征中，只需要简单的上采样就能恢复那些细粒度特征；

+ **Neck分析优化**：Neck的作用是对BackBone的特征进行聚合，进而增加感受野、融合多尺度的上下文信息，进而应对不同尺寸的目标。PillarNet、MVX-Net等架构都使用了多尺度特征融合的设计，对不同阶段的特征图进行上采样到同一分辨率后级联起来。该团队想搞清楚3D检测到底需不需要多尺度特征聚合，因此进行了一系列实验。基于实验获得结论：多尺度特征聚合不是必须的，单尺度的neck就可以表现得更好。因此给3D检测领域的neck设计打开一个思路：探索更多2D检测领域的优秀neck设计，将其移植到3D检测的neck上，说不定会有更大的提升。

+ **Head优化**：PillarNext采用CenterPoint的head机制，然后进行了特征上采样、multi-grouping和IoU Branch等优化，最终显著提高了检测指标。

PillarNext的整体结构如下图所示：

![image-20230523135257154](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230523135257154.png)

这篇文章想要颠覆已有点云3D检测框架，提出了新的基于pillar形式的点云3D检测框架，而且作者指明后续演进方向是将2D检测的优秀成果嫁接到基于pillar的3D检测领域。但是这篇文章的作者杨晓东在3D检测领域可以说是名不见经传的人。其先在Nvidia工作，后加入了轻舟智航，从业期间几乎没发表点云3D检测的相关工作。而且从文章中可以看出，该团队对点云3D检测的领域的的掌握不是很全面。所以直接甩出这样一篇文章，难免让人有所质疑。

但是抛开作者在PillarNext架构上的一系列设计不谈，这篇文章对编码方式、BackBone、neck等方面的一系列分析与对比还是很有价值的，这些分析数据和结论值得我们认真思考。

总之，这篇文章的作者做到了大胆假设，我们读者要小心求证。如果自行验证结果与论文观点一致，那么这篇文章就真的可以称为宝藏了，且让我们拭目以待。

# 横向对比与分析

本节对比4个框架的主要改优化点，通过横向对比来归纳和总结设计原则和规律。然后基于总结的原则和规律，就可以针对性地探索某一问题的具体优化方法。

这4个框架在各模块的优化方法如下表所示：

|             |                       VoxelNext                       |               PointNext                |                  PillarNet                   |            PillarNext             |
| :---------: | :---------------------------------------------------: | :------------------------------------: | :------------------------------------------: | :-------------------------------: |
| 3D BackBone |              3D稀疏卷积+下采样+voxel剪枝              |            MLP+SA+InvResMLP            |                    不涉及                    |              不涉及               |
| 2D BackBone |                    基于2D稀疏特征                     |                 不涉及                 | 4层2D Sparse CNN，下采样倍数分别为1、2、4、8 |      ResNet18 2D Sparse CNN       |
|    Neck     |                        未优化                         |             多尺度特征聚合             |            语义特征与空间特征聚合            |               ASPP                |
|   感受野    | 下采样层：3D 稀疏卷积BackBone加两层下采样来增加感受野 | 邻域半径：通过调整领域半径来增加感受野 |                    未优化                    |          不需要额外优化           |
|    Head     |                    max-pool代替NMS                    |                 未优化                 |                    未优化                    | CenterHead+上采样，恢复细粒度特征 |

下面我们分别对每个模块的设计原则的技巧进行分析：

## "next" 3D BackBone设计思路

3D BackBone可借鉴的思路不多，点云3D检测领域是3D BackBone的探路者。

从本文提到的4篇文章来看，只有VoxelNext和PointNext涉及3D BackBone。而PointNext的3D BackBone主要创新是提出InvResMLP，本质还是MLP的堆叠，不算是全新的涉及。对于VoxelNext来说，其3D BackBone没做结构上的优化，还是基于SECOND的3D BackBone结构，只是增加了下采样层和voxel剪枝操作。

从整个点云3D检测领域来看，3D BackBone主要分3类：

+ **基于MLP的3D BackBone**：PointNet系列都采用此结构，同时一些voxel-point结合的3D检测算法也用MLP组来提取point的特征，归根到底，这也是借鉴了PointNet的BackBone思路。

+ **基于3D稀疏卷积的BackBone**: 比如类似SECOND结构的BackBone, 当前基于voxel的3D检测框架大部分都采用此结构的BackBone。

+ **基于transformer的3D BackBone**: 基于Transformer的3D BackBone比较少，还有待学者们进一步提出更多更优秀的transformer BackBone, 目前的代表性方法有：

  + TransVoxel：这是一种基于Transformer的3D卷积神经网络，用于处理体素化的3D数据。TransVoxel利用Transformer的自注意力机制来处理3D空间中的长范围依赖性，从而捕获3D数据中的复杂模式。
  + SETR（SEgmentation TRansformer）：尽管SETR主要用于语义分割任务，但其在处理3D数据时展现出了潜力。SETR将点云数据转化为体素，然后使用Transformer进行特征提取。
  + Point Transformer：这是一个直接在点云数据上应用Transformer的网络，其引入了一种新的自注意力机制来处理无序的点云数据。
  + M3DETR：这是一个为3D目标检测设计的基于Transformer的模型，它利用Transformer来处理在3D环境中的目标关系。
  + DSVT：这是一篇用Transformer实现3D BakcBone的文章，整体框架基于CenterPoint实现，特点是精度高、易部署、算子依赖低。

  当前transformer在点云3D检测领域的应用大多还是各种形式的特征聚合，完全基于transformer的3D BackBone的研究还远远不够成熟，这方面的研究空间也很大。我们可以保持这方面的关注，同时也可以自己下场进行分析与实验，不吝贡献自己的力量来推动该领域的发展。

## "next" 2D BackBone设计思路

除point结构外，剩下3个基线框架在2D BackBone上的设计理念保持一致：即都从传统的2D稠密特征转换成基于稀疏特征的设计，只是各自采用的具体方法不同。为什么这几个团队都不约而同采用这种设计呢？他们这样设计的理由是什么？

+ **VGG（Visual Geometry Group）**：VGG网络是一种深度卷积神经网络，由牛津大学的Visual Geometry Group开发，特别是VGG16和VGG19在许多任务中都表现出色。
+ **ResNet（Residual Network）**：ResNet通过引入残差连接来解决深度网络中的梯度消失问题。ResNet有多个版本，包括ResNet-50、ResNet-101和ResNet-152。
+ **DenseNet（Densely Connected Convolutional Networks）**：DenseNet通过在每一层都使用前面所有层的输出作为输入，提高了网络的信息流通性。
+ **EfficientNet**：EfficientNet是一种通过自动搜索来优化网络结构的方法。EfficientNet-B0到EfficientNet-B7是一系列的网络，它们在保持计算效率的同时提供了优秀的性能。
+ **HRNet (High-Resolution Network)**：HRNet保持了高分辨率的特征图 throughout the whole process，这对于小目标的检测非常有帮助。
+ **ResNeSt**：ResNeSt是一种改进的ResNet网络，它引入了分割注意力机制（Split-Attention）来提高性能。
+ **RegNet**：RegNet是一种基于自动机器学习的网络，它通过学习网络的深度和宽度的规则来优化性能。
+ **ResNeXt**：ResNeXt引入了分组卷积的概念，这可以帮助网络学习更多的特征，对于处理遮挡和小目标的场景非常有用。
+ **Vision Transformer (ViT)**：ViT是一种将Transformer网络应用到视觉任务的方法。与传统的卷积网络不同，ViT直接在图像的像素级别上进行操作，显示出了优秀的性能。
+ **Swin Transformer**：Swin Transformer是一种改进的Vision Transformer，它引入了滑动窗口机制来处理局部信息，同时保留了Transformer的全局感知能力。
+ **PVT**：PVT是一种基于Transformer的多尺度视觉模型。PVT的主要创新点在于引入了金字塔结构，使得模型可以在不同的尺度上处理图像特征，这对于处理不同大小的目标非常有帮助。PVT在一些视觉任务上，如ImageNet分类和COCO目标检测，都取得了很好的结果。
+ **DeiT (Data-efficient Image Transformers)**：DeiT是一种改进的Vision Transformer，它通过更有效的数据使用和训练策略，使Transformer能够在相对较小的数据集上进行训练。
+ **CvT (Convolutional Vision Transformer)**：CvT结合了卷积神经网络（CNN）和Transformer的优点，它在Transformer的基础上引入了卷积结构，以更好地处理局部特征。
+ **TNT (Transformer in Transformer)**：TNT在每个Transformer的注意力模块中嵌入了另一个Transformer，以同时处理图像的局部和全局特征。
+ **CoaT (Co-Scale Conv-Attentional Image Transformers)**：CoaT引入了卷积和注意力的共尺度（Co-Scale）设计，以提高模型的性能和效率。

未来将上述经典的2D BackBone进行稀疏化适配可能会进一步提高点云3D检测算法的性能。

## "next" Neck设计思路

 "Neck"的作用是对BackBone输出的特征进行进一步的处理和转换，以提供更适合特定任务的特征表示。

具体来说，"Neck"模块通常由一些额外的卷积层或其他操作组成，例如池化、上采样、特征融合等。它可以降低特征的维度、压缩特征信息、引入上下文信息，从而为后续任务提供更丰富的特征表示。

本文涉及的"next"框架的neck优化思路大体一样：通过多尺度特征融合或语义特征聚合来达到特征增强的目的。VoxelNext没有提出新的neck技术，但是SECOND系列框架本身的neck就已经包含多尺度特征融合功能了。特别的是：PillarNext第一次完全将2D领域的Neck ASPP完整移植到3D检测框架上。ASPP是一种空洞卷积的空间金字塔池化结构，它可以在多个不同的尺度上提取特征，对于处理不同大小的目标非常有效。

按照PillarNext的思路，2D检测领域还有很多经典neck设计值得我们尝试或者借鉴，比如：

+ **FPN (Feature Pyramid Network)**：FPN是一种多尺度特征提取网络，它通过自顶向下的路径和横向连接，将高层的丰富语义信息和低层的高分辨率信息进行融合，对于检测不同大小的目标非常有效。
+ **PANet (Path Aggregation Network)**：PANet在FPN的基础上，引入了自底向上的路径增强，使得高层的特征可以得到更多的低层细节信息。
+ **NAS-FPN (Neural Architecture Search-Feature Pyramid Network)**：NAS-FPN使用神经结构搜索技术，自动搜索最优的特征融合结构，使得模型的性能得到了进一步的提升。
+ **BiFPN (Bidirectional Feature Pyramid Network)**：BiFPN在FPN的基础上，引入了双向的特征融合，使得信息可以在多个尺度之间自由流动。
+ **PSPNet（Pyramid Scene Parsing Network）**：PSPNet利用金字塔池化（Pyramid Pooling）来增加感受野。它将输入特征图分成多个子区域，并在每个子区域上进行池化操作，得到不同尺度的特征表示。然后，这些特征通过1x1卷积进行融合，以获取全局和局部上下文信息。通过这种方式，PSPNet能够捕捉到不同尺度的语义信息，提高图像分割任务的性能。
+ **DCNN（Dilated Convolutional Neural Network）**：DCNN通过在主干网络中使用空洞卷积来增加感受野。空洞卷积可以通过调整膨胀率（dilation rate）来扩展卷积核的感受野，从而在不引入额外参数和计算量的情况下增加感受野大小。DCNN的设计可以在保持特征图分辨率的同时，增加模型对更广阔上下文信息的感知能力。
+ **DANet（Dual Attention Network）**：DANet引入了双重注意力机制来提高感受野。它包括通道注意力和空间注意力两个注意力模块。通道注意力用于对通道维度上的特征进行加权，以提取重要的特征信息。空间注意力用于对空间维度上的特征进行加权，以捕捉特征之间的关联性。通过双重注意力机制，DANet能够增强模型对全局和局部信息的感知能力。
+ **SANet（Selective Attention Network）**：SANet通过选择性注意力机制来提高感受野。它引入了一个选择性注意力模块，该模块可以自适应地选择和加权输入特征图的不同通道。通过选择性注意力机制，SANet能够自动学习对目标有用的特征通道，并集中注意力在重要的特征上，以提高感受野和模型性能。

## "next" Head设计思路

这四篇文章在head上的优化不多，主要是VoxelNext由于全稀疏检测架构需要head做对应适配，PillarNext在CenterHead的基础上加了上采样来恢复细粒度几何特征。这其中可能是PillarNext的head上采样思路更具启发性。按照PillarNext的观点，细粒度的几何特征已经在BackBone里编码进特征里了，虽然经过很多层的下采样，但是只要在head里加入上采样就可以恢复这些细粒度信息。

此外，CenterPoint的head设计也是PillarNext的主要head构成，我们可以将CenterPoint的head思路和上采样恢复信息的思路运用到其他传统框架中，看看能否得出更好的效果。

## "next"训练技术

这4篇"next"框架中，只有PointNext对训练技术进行了详细的分析和优化，同上面的数据增强技术一样。但是PointNext的分析实验非常详细，为我们进一步优化训练策略提供了思路。

训练技术包括损失函数、优化器、学习率调度器(learning rate schedulers)、超参数。 对于上述几个优化技术，PointNet++使用的有：交叉熵损失函数、Adam优化器、exponential learning rate decay。 受益于机器学习理论的发展，现在的神经网络可以用更好的优化器，比如AdamW；还有更高级的损失函数，比如CrossEntropy with label smoothing。 与数据增强的验证方式一样，PointNext团队也对训练技术做了控制变量实验，实验结论是：CrossEntropy with label smoothing， AdamW, Cosine Decay在大部分场景下都表现优秀。

我们可以在此基础上做更广泛和更深入的分析。

## "next"数据增强技术

除了前面提到的BackBone、neck和head之外，数据增强也在点云3D检测中扮演重要角色。

这几篇"next"文章中，只有PointNext对数据增强做了细致的实验和分析，除了常用的旋转、缩放、jitter等，还使用了更新更先进的数据增强技术。比如KPConv里的randomly drops colors，Point BERT里的点云重采样策略、点云重采样、随机噪声、height appending、全场景输入、color drop、color auto contrast等。

实验数据如下：

![image-20230519170114194](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230519170114194.png)

这为我们打开数据增强方面的思路：我们也许可以尝试更多的数据增强技术，通过控制变量的方式逐个分析哪些数据增强技术是有效的。

此外，也可以把视野再放大一点，把LC融合算法中的数据增强技术移植到"next"系列框架，比如VirConv中删除近处的点云、保留远处点云的做法。另外，也可以将最近很流行的点云生成技术移植进来，比如PDA、BtcDet、GLENet的点云补全技术。





























