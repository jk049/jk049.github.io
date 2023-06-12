---
title: "3D检测论文解读——VoxelNext"
date: 2023-05-22T09:54:31+08:00
description: "全稀疏结构的voxel框架，想替代当前的SECOND结构，称为下一代voxel基线框架"
draft: false
author: "jk049"
cover: "https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/屏幕截图 2022-10-19 211941.png"
tags: ["next","point","3D","voxel"]
theme: "light"
---

# 一句话总结

这篇文章是港中文的贾佳亚团队、香港大学、旷世科技张翔雨团队合作的3D检测算法，主要创新点是：将经典的SECOND结构改为极简架构，以全稀疏的方式直接从voxel特征检测目标。实验指标也不错，在NuScene上的检测效果优于CenterPoint，劣于PillarNet，但是速度很快。

# 摘要

**背景**：3D检测算法总是依赖于anchor或center，并且大都是将2D框架转换为3D框架。因此需要将稀疏的voxel特征转换为稠密特征，然后用基于稠密特征的head输出检测结果。这些稠密数据导致计算量增加。

**本文方法**：本文提出完全基于稀疏数据的3D检测框架VoxelNext，该框架完全基于稀疏voxel特征进行检测，不必将稀疏特征进行其他转换或anchor、nms等操作。

**实验结果**：实现检测精度和检测速度的较好平衡，在NuScene上的跟踪指标最高，在WOD和AV2上也表现不错。 代码仓：https://github.com/dvlab-research/VoxelNeXt

# 基本介绍

**传统方法的问题**：最近主流的3D检测框架基本都基于3D稀疏卷积进行特征提取，然后借鉴Faster RCNN或CenterNet的思路设计head。这些方法有太多的人工设计和数据转换，数据处理不够直接，架构设计不够优雅，计算效率不够高。

anchor和center机制原本是为图像这样的规则数据设计的，没考虑3D数据的非规律性和稀疏性。为使用anchor和center机制，主流方法需要将3D稀疏特征转换为2D稠密特征，然后接上述的检测头。

这些方法尽管取得了不错的检测效果，但是也引入了问题，比如计算效率低、结构复杂等。

下面举例具体说明： ![image-20230522094804366](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094804366.png)

上图中的CenterPoint的热力图可显示大部分空间的预测分数为0，占比约1%。然后转换为稠密特征后，需要在所有位置进行运算，这一方面增加了计算量，另一方面导致重复检测，又引入nms之类的后处理机制删除多余检测目标。

**本文方法**：为进一步优化3D检测框架，该团队提出VoxelNext结构，该方法简单高效、不需要后处理。本算法的核心是voxel-object机制，统统稀疏卷积网络直接从voxel特征检测目标。

本框架的优点是简单，没有稀疏特征到稠密特征的转换，不需要anchor、proposal和nms等复杂的组件。整体结构如下图所示：![image-20230522094826144](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094826144.png)

**其他全稀疏方法**：最近FSD也采用了全稀疏的框架，然后基于VoteNet的投票机制对proposal进行refine。又因为点云的分布可能不能完全表征目标形状，所以上述的投票机制需要不断迭代来提高检测精度。FSD的计算效率不如该团队的VoxelNext,  对比数据如下图所示：![image-20230522094846644](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094846644.png)

# 相关工作

**voxel转2D框架**：3D检测框架大多与2D检测框架类似，比如RCNN系列的PV-RCNN、Voxel RCNN、Pyramid RCNN、Graph RCNN等；还有CenterPoint系列的CenterNet、Center Point等。

3D检测与2D检测问题的不同点在于点云的稀疏性，但是很多方法的检测头还是基于2D稠密卷积。

比如VoxelNet借鉴PointNet的方法提取voxel特征，然后转到2D结构；SECOND将VoxelNet中的voxel特征提取方法改为3D稀疏卷积。后来的SOTA方法基本都是沿用此结构。

**voxel转热力图框架**：CenterPoint将voxel特征转换为稠密特征，然后预测热力图预测目标质心。

此类方法被一些BEV融合方法借鉴，比如BEVFusion系列。

**稀疏检测框架**：之前提出的稀疏检测框架虽然方法各不相同，但是都很复杂。典型方法如下：

- RSN在range图像上使用前景分割，然后基于分割的稀疏数据进行3D检测；
- SWFormer结合稀疏transformer、分割窗口、multi-head和feature pyramid机制；
- FSD使用点云聚类解决目标中心特征丢失问题；

**３D卷积网络**：３D稀疏卷积提取３D特征效率很高，但是对于检测来说，提供的信息不够。

所以常用的３D检测框架都将３D稀疏特征转换成２D稠密特征，来对特征进行加强。

还有一些方法对稀疏卷积网络进行优化的，比如：

- Focal　Sparse　ConvolutionａｌNetworks　for　３D　Object　Detection
- Spatial　Pruned　Sparse　Convolution　for　Efficient　３D　Object　Detection

还有一些方法用transformer替代３D稀疏卷积，比如

- Voxle　Set　Transformer：A　Set　ｔｏ　Set　Approach
- VoTr

针对上述的感受野不足以及各种解决办法，该团队论证了**只需要增加下采样层**就可以解决，无需其他复杂设计。

**３D目标跟踪方法**：大多数MOT方法都直接基于检测结果进行卡尔曼滤波，CenterPoint预测目标速度后进行关联，然后复用CenterTrack结构。

# VoxelNet设计细节

点云或者voxel在目标表面的分布基本都是不规律的、离散的。因此该团队想基于voxel提出一个不用anchor的检测框架。

该团队希望基于当前的３D稀疏卷积进行最小修改，实现基于voxel直接进行目标检测。本框架主要包含３个创新点：BackBone，稀疏检测头和跟踪模块。下面分别进行详细介绍：

## 稀疏卷积BackBone的适配

基于稀疏的voxel特征直接进行精确的3D检测要求特征必须表征能力强且感受野够大。

虽然3D稀疏卷积特征用于3D目标检测已经持续很多年了，但是最近的研究表明其特征信息不够充分，需要用Focal Sparse Convolution、Transformer等机制进行进一步增强。

3D BackBone整体结构如下图所示：

![【VoxelNext 3D BackBone】.excali](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/【VoxelNext 3D BackBone】.excali.svg)

其中，红色背景模块为3D BackBone的修改部分，包括下采样层的Conv4/Conv5模块，高度压缩对应的Bev_out模块，voxel剪枝对应的DynamicFocalPruningDownsample模块。下面将具体介绍这三个改动点。

### **增加下采样层**

**设计理论**：针对此问题，该团队的解决办法是增加一个下采样层。 一般来说，3D稀疏卷积包含4层，对应的特征步长分别为1、2、4、8；这种结构产生的特征不足以进行航向预测。为对特征进行增强，该团队对原结构增加两个下采样层，对应的步长是16和32。

这个小改动极大地增加了感受野，效果示意图如下所示：![image-20230522094911441](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094911441.png)

**具体实现**：结合BackBone结构图，我们可知论文并没有将下采样层的所有设计细节呈现出来。从实现方面看，BackBone下采样层总共有3个修改点：

+ 下采样层：论文里所说的增加两层下采样层，在代码里的实现方式是增加Conv5和Conv6两个稀疏卷积模块。这两个卷积模块的输入输出通道数都是128，与Conv4一样，这样设计是为了下一步的特征级联。

+ 特征相加：特征级联的目的是将Conv5和Conv6的特征加到Conv4的对应位置上。由于Conv5和Conv6分别进行了步长为2的卷积下采样，所以特征相加之前要将Conv5和Conv6的位置进行恢复。特征相加的代码如下：

  ```    python
  x_conv5.indices[:, 1:] *= 2
  x_conv6.indices[:, 1:] *= 4
  x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
  x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
  ```

+ Shared_Conv: Shared_conv是卷积核为3，输入输出通道为128的submanifold稀疏卷积模块，论文里似乎没说这个所谓的共享卷积层是干什么的。我们且往下看。

### **高度压缩**

典型3D检测框架将3D特征转换为2D稠密特征，但是该团队发现2D稀疏特征就足以进行3D检测。因此该团队将voxel特征投影到平面上，相同位置的特征相加。高度压缩代码如下：

```python
def bev_out(self, x_conv):
    features_cat = x_conv.features
    indices_cat = x_conv.indices[:, [0, 2, 3]]
    spatial_shape = x_conv.spatial_shape[1:]

    indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
    features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
    features_unique.index_add_(0, _inv, features_cat)

    x_out = spconv.SparseConvTensor(
        features=features_unique,
        indices=indices_unique,
        spatial_shape=spatial_shape,
        batch_size=x_conv.batch_size
    )
    return x_out
```

【问】逻辑不通呀，3D特征不够，求和成2D特征就够？

### **voxel剪枝**

**设计理论**：3D点云包含大量背景点，对目标检测没用。该团队用下采样层不断裁剪没用的voxel，按照VS-Net中SPS-Conv的机制，该团队有效控制了voxel数量。示意图如下所示![image-20230522094928753](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094928753.png)

**具体实现**：voxel剪枝模块模块以降低检测精度为代价来提高推理速度。从实现来看，该模块比较复杂，而且效果有限，可能有更好的选择，比如VirConv的近端下采样方式。因此本文不对VoxelNext的voxel剪枝细节展开，只简单告诉读者有这个东西，类的定义在DynamicFocalPruningDownsample类中，基本结构如下图所示：

![image-20230529155917347](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230529155917347.png)

**实验数据**：经过实验分析，大概可以减少一半的voxel，数据如下表：![image-20230522094947274](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522094947274.png)

## 稀疏检测头

该团队直接基于3D稀疏特征进行检测，训练时，将voxel assign到最近的真值框。损失函数使用Focal Loss。

**voxel选择**：该团队观察到，query voxel通常不在目标中心，甚至不在目标框内。关于NuScene数据集中query voxel 与目标框的位置关系统计信息如下表所示：![image-20230522095007697](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095007697.png)

此外，推理时该团队使用max-pooling来代替NMS。与Submanifold稀疏卷积类似，max-pooling只在非空地方执行，从而节省了head的计算量。

**box回归**：按照CenterPoint的原则，该团队对平面位置、高度、朝向角进行回归；此外对于NuScene数据集或跟踪任务，该团队还回归速度信息。这些预测结果的损失函数基于L1损失函数。

对于Waymo数据集，该团队对于IOU的预测通过kernel size为3的submanifold卷积层。该方式比全连接层精度高，时间消耗差不多，具体数据如下：![image-20230522095026161](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095026161.png)

VoxelNext的head结构如下图所示：

![【VoxelNext Head结构】.excalidraw](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/【VoxelNext Head结构】.excalidraw.svg)

## 3D跟踪

跟踪模块是基于CenterPoint的扩展，基于速度对目标位置进行预测，基于L2距离计s算关联关系。示意图如下所示：

![image-20230522095045639](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095045639.png)

# 实验结果

下采样层的消融实验如下表所示：![image-20230522095105421](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095105421.png)

voxel剪枝的消融实验如下表：![image-20230522095128793](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095128793.png)

![image-20230522095144582](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095144582.png)

本文选择的剪枝比例是0.5。

高度压缩的消融实验：

如果BackBone和head都用3D稀疏卷积的话，计算量太大，因此将检测头设计成基于2D的。

![image-20230522095204068](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095204068.png)

在各数据集上的检测指标如下：

![image-20230522095220122](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095220122.png)

![image-20230522095232232](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095232232.png)

![image-20230522095243299](https://blog-pic-bkt.oss-ap-southeast-1.aliyuncs.com/img/image-20230522095243299.png)

# 个人分析于总结

**作者信息**：这篇文章是港中文的贾佳亚团队和旷世科技的张翔雨团队合作的。两位都是领域大佬，贾佳亚大佬之前提出很多点云3D检测方法，比如IPOD、3DSSD、STD等，张翔雨大佬之前提出过LC融合3D检测方法PETR。值得注意的是，贾佳亚团队之前提出的3D检测算法都结构复杂、计算量大、难以商业落地，但是这次提出的VoxelNext主打一个简单快速好落地。这个模型设计上的思路转变还是挺大的，至于所提设计是否有效，且让该团队拭目以待。

**创新点**：提出基于voxel的稀疏检测框架。之前Voxel框架都基于SECOND结构，先3D稀疏卷积，然后转成2D稠密特征后进行2D卷积，再接proposal、head那一套流程。VoxelNext的全稀疏voxel框架砍掉了后面2D的那一连串流程，大大简化了结构、节省了时间。

**大话演进方向**：上面的内容都是基于论文内容的客观陈述，接下来是个人主观分析，请读者持辨证的态度阅读。分析基于个人掌握的知识，从本文出发，对3D检测技术的演进方向进行主观分析，因此称为“大话”。但是后续我会进行独立实验及相关的研究跟踪，预知后续分析结果，请保持关注。

话不多说，上菜：

- 与LC融合方法结合：当前的LC融合方法越来越成熟，检测指标也逐渐超越了基于单Lidar的检测算法。但是目前主流的LC融合方法的L分支还是基于SECOND结构，即voxel特征转2D的结构。此文的方法如果有效，则可以对目前所有基于SECOND结构的3D检测方法进行框架升级，从而提高检测精度，并且大大减少推理时间。这个方向是该文章最具吸引力的演进方向，该团队可以尝试做点实验进一步验证；
- 与其他全稀疏检测技术结合：该文章主打的还是基于voxel形式的全稀疏框架，但是最近也出现一些其他的全稀疏检测技术，而且效果不错。VoxelNext可以尝试与其他全稀疏技术进行碰撞，说不定会有更精彩的火花；
- 嫁接到其他优秀的3D检测框架上：有些优秀的3D检测框架是在基于SECOND结构的框架做加法，比如MPPNet引入连续帧序列的方法提高检测精度、GLENet对输入数据做文章。可以将那些优秀尝试移植到VoxelNext框架上，看看能开出什么花来；
- 与其他”next”辈的框架融合：22年、23年的点云3D检测算法的演进有这样的趋势：基于经典的SECOND、PointPillars、PointNet框架几乎没有新的演进，仿佛经典框架已经触到天花板，没什么优化空间了。于是22年、23年的点云3D检测算法四面突击：有的尝试对输入数据做文章，比如SPG、PDA、BtcDet、GLENet等；有的尝试引入序列检测的技术，比如MPPNet；有的引入全稀疏概念，比如SST、SFD等；还有的尝试对经典框架进行升级，提出各种”next”概念，比如PointNext、PillarNext，以及PillarNet等。VoxelNext可以与其他next思路结合，取长补短，从而推出更高效、更有效的“next”基线框架；