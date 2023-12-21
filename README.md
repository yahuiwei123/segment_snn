## Segment_SNN
### 概览
Segment_SNN主要实现了一个利用VGG16和FPN作为backbone，采用encoder-decoder架构实现的语义分割网络。其中对VGG16和FPN网络实现了细粒度的脉冲神经元替换（卷积、线性、池化、批归一化）。
我们的实验主要分为两个阶段
+ 首先是对VGG16网络进行SNN的转换，并将转换后的网络在nminst数据集上进行分类训练，来验证模型转换效果
+ 第二阶段我们对FPN网络也进行了转换，并且将其与上一步转换过的VGG16网络进行拼接得到我们的语义分割模型Segment_SNN

文件内容：
+ 数据文件夹结构
  + datasets
    + coco
      + val2017
+ basic.py文件包含了所有的神经元以及基础模块的定义
+ dataset.py文件定义了数据的预处理方式（包含快速眼动生成序列帧以及数据增强）
+ fpn.py文件包含转换后的脉冲FPN网络
+ vgg16.py文件定义了转换后的脉冲VGG16网路
+ model.py定义了整合VGG和FPN网络的分割模型Segment_SNN
+ train.py包含训练代码
+ predict.py包含预测代码
### 使用方式
#### 数据集准备
+ 使用coco数据训练
+ 数据使用快速眼动法生成dvs帧（共9帧）作为模型输入，方法参考`Lin Y, Ding W, Qiang S, et al. Es-imagenet: A million event-stream classification dataset for spiking neural networks[J]. Frontiers in neuroscience, 2021, 15: 1546.`

#### 模型训练
```python
python train.py --batch_size 8 --step 8 --learning_rate 0.01 --num_epochs 100 -output_size (480, 480)
```

#### 模型预测
```python
python predict.py --image_path './test/img' --step 8 --output_size (480, 480) --output_dir './test/out'
```
### 模型细节
+ 快速眼动法生成dvs数据集
  + 每张图片通过移动差分生成8张差分序列（包括原数据共9帧），差分后的dvs数据包括正负两通道，在通道维拼接（6通道），输入到模型中。
+ 对模型中的如下模块进行了替换
  + Conv2D $\rightarrow$ LayerWiseConvModule
  + Linear $\rightarrow$ LayerWiseLinearModule
  + BatchNorm $\rightarrow$ TEP
+ 对模型使用不同种类神经元类型进行了实验
  + BiasLIFNode $\rightarrow$ DoubleSidePLIFNode（通过将初始x与avgpool(x)统一维度后做差完成正负脉冲的实现）
### 实验效果展示
#### 第一阶段（分类）
+ 我们在nminst数据集上训练17个epoch后的结果
<img width="503" alt="94d9608c9238ed6bfae8465e9da21d9" src="https://github.com/yahuiwei123/segment_snn/assets/84215971/99bc2e72-d151-4a2b-bdce-2b81c9982185">

#### 第二阶段（分割）
由于计算资源有限，仅使用小样本数据集进行训练，数据集分布不均匀，因此模型收敛速度较慢，以下是相应的训练结果。
+ 训练的Kaggle Notebook（包括FCN和SNN）：https://www.kaggle.com/code/littleweakweak/test-pycocotools
后续在增大数据集规模后，可考虑进一步改进模型，以获得更好的训练效果。

### 成员分工
+ 韦亚辉：VGG分割模型向Snn模型转化 模型训练
+ 徐一翀：数据预处理 基于快速眼动算法的dvs数据集转化 模型训练
+ 王妍紫：基于神经元的Snn模型改进 模型训练
+ 袁冶：数据预处理 基于快速眼动算法的dvs数据集转化
+ 陈浩林:相关文献阅读 关键理论支撑与准备工作
