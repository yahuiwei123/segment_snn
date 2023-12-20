### Segment_SNN
#### 概览
Segment_SNN主要实现了一个利用VGG16和FPN作为backbone，采用encoder-decoder架构实现的语义分割网络。其中对VGG16和FPN网络实现了细粒度的脉冲神经元替换（卷积、线性、池化、批归一化）。
#### 使用方式
##### 数据集准备
+ 使用coco数据训练
+ 数据使用快速眼动法生成dvs帧（共9帧）作为模型输入，方法参考`Lin Y, Ding W, Qiang S, et al. Es-imagenet: A million event-stream classification dataset for spiking neural networks[J]. Frontiers in neuroscience, 2021, 15: 1546.`
+ 如果使用个人数据集，确保包含如下目录和文件
  + xxx
  + xxx
##### 模型训练
```python
python train.py --batch_size 8 --step 8 --learning_rate 0.01 --num_epochs 100 -output_size (480, 480)
```

##### 模型预测
```python
python predict.py --image_path './test/img' --step 8 --output_size (480, 480) --output_dir './test/out'
```
#### 模型细节
+ 对模型中的如下模块进行了替换
  + Conv2D $\rightarrow$ LayerWiseConvModule
  + Linear $\rightarrow$ LayerWiseLinearModule
  + BatchNorm $\rightarrow$ TEP
+ 对模型使用不同种类神经元类型进行了实验
  + BiasLIFNode $\rightarrow$ DoubleSidePLIFNode（通过将初始x与avgpool(x)统一维度后做差完成正负脉冲的实现）
#### 实验效果展示

#### 成员分工
