# 飞桨常规赛：遥感影像地块分割 - 9月第4名方案



## 项目描述：

​     赛题旨在对遥感影像进行像素级内容解析，并对遥感影像中感兴趣的类别进行提取和分类，以衡量遥感影像地块分割模型在多个类别（如建筑、道路、林地等）上的效果。

本项目使用了Paddle提供的PaddleSeg中的deeplabv3+模型，进行了微调，并修改损失函数为diceloss。

> PaddleSeg是基于飞桨[PaddlePaddle](https://www.paddlepaddle.org.cn/)开发的端到端图像分割开发套件，涵盖了**高精度**和**轻量级**等不同方向的大量高质量分割模型。通过模块化的设计，提供了**配置化驱动**和**API调用**两种应用方式，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。

PaddleSeg的使用可以参考[PaddleSeg: End-to-End Image Segmentation Suite Based on PaddlePaddle. (『飞桨』图像分割开发套件） (gitee.com)](https://gitee.com/paddlepaddle/PaddleSeg?_from=gitee_search)

## 项目结构

```shell
|--benchmark
|--configs
|--contib
|--data                      # 数据集路径，数据集组织方式如下，txt文件已经提供
|  |--img_testA
|  |--img_train
|  |--lab_train
|  |--train.txt
|  |--val.txt
|--deploy
|--docs
|--legacy
|--outputs                   # 模型结果的输出路径
|  |--deeplabv3p_5
|     |--best_model
|        |--model.pdparams
|--paddleseg
|--script
|--slim
|--tools
|--.pre-commit-config.yaml
|--.style.yapf
|--.travis.yml
|--export.py
|--predict.py                # 预测
|--README.md
|--requirements.txt 
|--setup.py
|--train.py                  # 训练
|--train_test.sh
|--val.py                    # 验证
```

## 训练参数：

```shell
batch_size: 4
iters: 80000

train_dataset:
  type: Dataset
  dataset_root: ../data
  train_path: ../data/train.txt
  num_classes: 4
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
    - type: Normalize
  mode: train


val_dataset:
  type: Dataset
  dataset_root: ../data
  val_path: ../data/val.txt
  num_classes: 4
  transforms:
    - type: Normalize
  mode: val

model:
  type: DeepLabV3P
  backbone:
    type: ResNet50_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz
  num_classes: 4
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]

optimizer:
  type: sgd
  weight_decay: 0.00004

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.001
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: DiceLoss
  coef: [1]
```



## 使用方式

##### 1.Requirements

-    Paddle版本：PaddePaddle2.1.2

-    python：python 3.7.5

- #####    数据集：[常规赛：遥感影像地块分割](https://aistudio.baidu.com/aistudio/datasetdetail/77571)，数据集请按照项目结构中data的子文件夹形式整理。

-    运行配置文件为：benchmark/mydeeplabv3p.yml，请根据实际情况对配置文件进行修改。

##### 2.生成分割结果

  提交时所使用的checkpoint，下载链接：https://pan.baidu.com/s/11QYDnB5W7n1WlvhNiiXX9w  提取码：d8bn

  下载后请放置在 ` outputs/deeplabv3p_5/best_model`下，并按所给的项目组织方式组织。

  命令行输入以下命令可得分割结果，结果保存在 `outputs/deeplabv3p_5/result/pseudo_color_prediction`下

```shell
python predict.py --config benchmark/mydeeplabv3p.yml --model_path outputs/deeplabv3p_5/best_model/model.pdparams \
                  --image_path ../data/img_testA --save_dir outputs/deeplabv3p_5/result
```

##### 3.训练

训练命令中包含验证命令，每1500验证一次，保留最好的模型。

```shell
python train.py  --config benchmark/mydeeplabv3p.yml --save_interval 1500 --save_dir outputs/deeplabv3p_5/ --do_eval
```

##### 4.预测

```shell
python predict.py --config benchmark/mydeeplabv3p.yml --model_path outputs/deeplabv3p_5/best_model/model.pdparams \
                  --image_path ../data/img_testA --save_dir outputs/deeplabv3p_5/result
```

##### 5.训练+预测，一键化脚本

```shell
sh train_test.sh
```

预测结果保存在 `outputs/deeplabv3p_5/result/pseudo_color_prediction`下

