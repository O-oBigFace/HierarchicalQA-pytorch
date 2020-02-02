# HierarchicalQA-pytorch 运行指南
论文[Hierarchical Question-Image Co-Attention for Visual Question Answering](http://papers.nips.cc/paper/6202-hierarchical-question-image-co-attention-for-visual-question-answering)的pytorch实现。

## 需要的环境（暂时）
> python3
>
> torch7
>
>luajit

## 数据准备
### 数据集
本模型使用的数据集为**cocoqa**。首先，在[Toronto COCO-QA Dataset](http://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/)上下载cocoqa数据集并解压，在项目根目录创建`data`文件夹，将解压得到的`cocoqa-2015-05-17`文件夹置入其中。
接着，到[coco](http://cocodataset.org/#download)下载图片数据，只需要下载cocoqa用到的*train2014*和*test2014*，并解压到任意文件夹。

### VGG19模型
这里我们要下载预训练好的VGG19模型：
下载地址(两个都需要)：[caffe model](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel), [prototxt](https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt)
在项目根目录创建文件夹`image_model`，将以上两个文件置入。

## 数据预处理
数据预处理需要切换到`prepro`文件夹：
```bash
cd prepro
```
### 处理cocoqa的数据
```bash
python cocoqa_raw_process.py
python cocoqa_prepro.py
```

### 利用caffe的VGG处理图像数据（待修改为python版本）
python版的caffe安装屡次失败，所以这里使用了torch7版本的caffe。
假设安装了luajit：
```bash
luajit img_process_vgg.lua --image_root [数据集coco的路径]
```
这一步需要时间可能会很长，具体由你GPU的算力决定。

运行结束后，数据处理部分的代码也就完成了。

## 模型训练（更新中）
回到项目根目录。
### 更改训练参数
使用如下命令可查看可更改的训练参数：
```bash
python train.py -h
```

### 模型训练
直接运行`train.py`：
```
python train.py [指定的参数]
```

### 查看结果
训练时，系统会自动将运行结果最优的模型存储起来，并且存储整个过程的loss以及accuracy。可在`save`文件夹中查看。