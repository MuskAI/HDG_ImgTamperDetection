# 前言
所有的数据集生成代码都在这里

# 实施计划

预计40天完成
一周完成数据集生成代码
两周完成模型代码和完善数据集生成代码，进行初步训练
两周完成参数调优和模型改进
两周完成5个对比实验，帮助完成文档

## 数据集
1. 利用coco数据集生成（最好使用两种数据集，因为coco不是逐项素标注,先实现coco数据集，使用coco train2017， ➡️50%）
2. 基于原本的拼接数据生成办法，生成篡改区域（✅）
3. 根据实际情况看看是否还需要改进数据集生成办法

## 模型
1. 将unet作为backbone，调研attention相关论文考虑加入注意力机制的问题

## 模型部署
Django(!!) 
Flask(easy to use)

## 流程
1. 生成数据集
2. 加载数据
   1. datasets：输入一张图image（3，320，320），（5，3，320，320）
   2. dataloader（输入是datasets）
3. 模型
   1. aspp 必须用上
4. train
   1. lr learning rate： 1e-2  1e-6 ，-7
   2. lr scheduler
   3. 交叉熵loss仔细看看
5. test

## 下周
1. srm aspp dilate 
2. blur 高斯噪声 （coco_SP COCO_CM + voc--> cod10k）

## 下次（12）
1.attention
2.扩充数据集
3.边缘引导（重点 双边padding 边缘条带）
4.指标 （coverage Columbia casia v2）


## 推荐
同济子豪兄
跟李沐学AI


## 下次
1. 效果再提升（）
2. coverage casia columbia 数据集 realistic（难度很大，不打算用）：和别人算法对比
3. coco合成数据：消融试验（unet unet+attention ours）
4. 前后端（延后）


## 最终模型结构说明
模型结构图, 三个loss函数，(分类loss 区域loss 和band loss)下面我来一一解释

分类分支的主要目的有两个： 
   1. 是用弱监督的方法，可以使用大量image-level标签的数据集，比如in the wild ， ps-battle.我们发现我们能在合成数据集得到非常好的效果，在一些公开数据集效果也不错，但是到真实情况下效果就非常差，所以希望通过弱监督的方式增加编码器的能力
   2. 在实际应用和使用中分类的标签很重要，这是偏实际的考虑

区域分支的主要目的有三个：
1. 我们的网络目的就是检测篡改区域，但是在试验过程中发现一些问题：
   1. 误检率比较高，倾向于把没有篡改的图片判断为有篡改，这也是mantranet等论文的问题，在19年zhou peng 的论文（GSR NEt）中也给出原因，因此要抑制语义信息的学习
   2. 对boundary检测结果比较差，对边缘附近往往分不清楚，呈现出来的效果是置信度比较低
   3. 从这两个问题出发，我们改进了loss函数，并添加了boundary监督分支
2. 改进区域监督的loss函数：我们在WCE 加权交叉熵的基础上，特别的对边缘附近进行加权，使之能够更好的处理边缘 

boundary分支的主要目的有两个：
1. boundary监督分支：为了进一步抑制对语义信息的干扰，让网络更加聚焦于局部细节信息（拼接篡改boundary），因此我们将copy-move和splicing都当作一类任务，如果没有这个分支，对copy-move的检测会出现大量误判
2. 我们认为CNN在学习映射上是擅长的（two stream那篇文章说的），因此我们只设计了一个与区域输出模块一样的模块来进行boundary的检测，希望网络的前三个解码阶段能同时提供篡改区域检测和篡改边缘检测的信息


## 需要的东西
1. band 效果图
2. 总的模型效果图，指标

coco voc，
Salzburg Texture Image Database (STex)（纹理数据）：coverage上，对相似性的纹理处理不好，所以要生成这种：同图篡改

1. 背景
2. 相关工作
3. 介绍算法
