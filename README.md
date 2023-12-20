### Segment_SNN
#### 概览
Segment_SNN主要实现了一个利用VGG16和FPN作为backbone，采用encoder-decoder架构实现的语义分割网络。其中对VGG16和FPN网络实现了细粒度的脉冲神经元替换（卷积、线性、池化、批归一化）。
#### 使用方式
##### 数据集准备
+ 使用coco数据训练
+ 如果使用个人数据集，确保包含如下目录和文件
  + xxx
  + xxx
##### 模型训练
``
python train.py --batch_size 8 --step 8 --learning_rate 0.01 --num_epochs 100 -output_size (128, 128)
``

##### 模型预测
``
python predict.py --image_path './test/img' --step 8 --output_size (128, 128) --output_dir './test/out'
``
#### 模型细节
+ 对模型中的如下模块进行了替换
  + Conv2D $\rightarrow$ LayerWiseConvModule
  + Linear $\rightarrow$ LayerWiseLinearModule
  + BatchNorm $\rightarrow$ TEP
+ 对模型使用不同种类神经元类型进行了实验
#### 实验效果展示

#### 成员分工