# 配置文件
> @author kuroshika  
> @E-mail huanglei701@126.com  
> @Date 2023/12/01 

## 文件结构

config文件目录下有mmfi，ntufi，ut等文件夹，每个文件夹代表一个数据集，
在每个数据集下有不同模型的配置文件。

## 配置文件说明

### 公有配置
在配置文件中，主要有如下一些部分组成：

- data_param 数据集的参数配置
- model_param 模型参数配置
- head_param 任务头参数配置
- optimizer_param 优化器设置
- training_param 训练设置
- 其他参数
  - debug 这个参数在调试的时候开启，不会新建日志，而是会在debug日志中输出
  - output_path 这个参数指定输出路径，不指定时会生成默认路径
  - pretrained_model 加载预训练模型的路径
  - device_id cuda设备
  - cuda_visible_device 可见的GPU
