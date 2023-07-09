一个暂时的readme

*模型功能*

1. 识别并分割原图中的人像（支持单人和多人）
  
2. 保留背景，并对原图中的人像进行风格迁移


*模型结构*

<img width="666" alt="截屏2023-07-05 上午10 04 58" src="https://github.com/DrXin2002/dachuang-final/assets/131842894/3c6f11ba-c4b3-4eef-90a0-019d7398563b">



*关于如何快速上手跑模型*

  ##必读##
  
1. 先在terminal中运行这样一段代码：\
   sudo chmod -R 777 .  \
   sudo chmod -R 777 [path/to/interpreter]\
   请将[path/to/interpreter]替换成自己的编译器路径，例如我的是：/Users/zhujiaxin/miniforge3/envs/studydeeplearning/bin/python

2. 将原图放入./input，请使用jpg、png、jpeg等常用格式\
   P.S.如果运行设备为mac，有的时候图片不会显示后缀名。
       这样的图片模型是无法读出的，请将其发送到微信传输助手后转存，即完成“无后缀”→“.jpg”的转换
   
3. windows用户需修改./AnimeGANv2/test.py以及./Mask_RCNN/predict.py，将其中的device修改为设备对应的device。\
   （windows设备对应device我都留了注释，只要把AnimeGANv2中的‘mps’和Mask_RCNN中的'cpu‘替换掉就可以）

4. 阅读./Mask_RCNN/readme.md，并根据指示下载模型预训练权重文件放入./Mask_RCNN/save_weights

5. 修改run_all.py中的interpreter_path（参考1.）
   
6. 运行run_all.py，等待模型return后即可在./result文件夹中找到最终生成结果。

  ##选读##
  
6. 如需调试单个模型，则需根据./AnimeGANv2/test.py以及./Mask_RCNN/predict.py中的长“#”分割线备注，注释掉相应语句，再对模型对应的test.py或者predict.py进行运行。

7. 文件的保存路径在test.py和predict.py较为靠前的位置，找不到可搜索关键字“dir”，由对应路径可找到模型中间产物的对应位置，方便调试。


  ##模型效果##（图片加载要等一下）
![WechatIMG135](https://github.com/DrXin2002/dachuang-final/assets/131842894/cd114bee-9fb6-45c2-8d40-3ad685c06e6b)

![WechatIMG136](https://github.com/DrXin2002/dachuang-final/assets/131842894/4e7f1d3a-3bd4-4777-9ffa-e56ab0ceff5a)

![WechatIMG133](https://github.com/DrXin2002/dachuang-final/assets/131842894/74e84d04-4331-443b-bed3-524dfd8bd045)

![WechatIMG134](https://github.com/DrXin2002/dachuang-final/assets/131842894/ad9339c9-88a2-43fa-a122-bbd6a8ef0e82)


[体现背景保持]  
|  原图  |  风格化全图  |  背景保持的人像风格迁移  |
![IMG_9987](https://github.com/DrXin2002/dachuang-final/assets/131842894/0bb5ce4e-d95b-4faf-b8ed-eecdfc2b6358)







