# Data-Mining-Project
## 搜索引擎返回结果的聚类

### 问题描述

#### 1、问题背景及分析

我们常常在百度上检索信息，但总是苦于搜索引擎不能准确地找到我们想要的信息，如何使搜索引擎的用户快速找到想要的结果呢？

文本聚类是一个将文本集分组的全自动处理过程，是一种典型的无指导的机器学习过程。类是通过相关数据发现的一些组，类内的文本和其它组相比更为相近。换一种说法就是，文本聚类的目标是找到这样一些类的集合，类之间的相似度尽可能最小，而类内部的相似性尽可能最大。作为一种无监督的机器学习方法，聚类是不需要训练过程的，也不需要预先对文档进行手工标注类别，所以聚类技术很灵活并有较高的自动化处理能力，目前已经成为对文本信息进行有效地组织、摘要和导航的重要手段，被越来越多的研究人员所关注。

如果能先对搜索引擎返回的结果进行文本聚类，并告诉用户每个类别的关键词，用户就能更快地从对应的类中找到想要的信息。

#### 2、问题描述

2.1 数据准备

爬取搜索引擎的搜索结果。

2.2 准备采用的方法或模型

首先进行数据预处理，然后对文档进行分词，使用BOW词袋模型将文档表示成向量，用kmeans、层次聚类等聚类算法对文档进行聚类，最后对文档和聚类结果进行可视化。

2.3 预期的挖掘结果

预期可以得到搜索引擎返回结果的聚类结果，用户可根据该结果在对应的类中搜索想要的信息。

### 项目评估

本项目使用轮廓系数评价聚类结果。

