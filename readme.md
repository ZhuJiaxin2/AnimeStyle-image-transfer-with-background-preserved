一个暂时的readme：关于如何快速上手跑模型

  ##必读##
1. 先在terminal中运行这样一段代码：
   sudo chmod -R 777 .
   sudo chmod -R 777 [path/to/interpreter]
   请将[path/to/interpreter]替换成自己的编译器路径，例如我的是：/Users/zhujiaxin/miniforge3/envs/studydeeplearning/bin/python

2. 将原图放入./input，请使用jpg、png、jpeg等常用格式
   P.S.如果运行设备为mac，有的时候图片不会显示后缀名。
       这样的图片模型是无法读出的，请将其发送到微信传输助手后转存，即完成“无后缀”→“.jpg”的转换
   
3. windows用户需修改./AnimeGANv2/test.py以及./Mask_RCNN/predict.py，将其中的device修改为设备对应的device。
   （windows设备对应device我都留了注释，只要把AnimeGANv2中的‘mps’和Mask_RCNN中的'cpu‘替换掉就可以）

4. 修改run_all.py中的interpreter_path（参考1.）
   
5. 运行run_all.py，等待模型return后即可在./result文件夹中找到最终生成结果。

  ##选读##
6. 如需调试单个模型，则需根据./AnimeGANv2/test.py以及./Mask_RCNN/predict.py中的长“#”分割线备注，注释掉相应语句，再对模型对应的test.py或者predict.py进行运行。

7. 文件的保存路径在test.py和predict.py较为靠前的位置，找不到可搜索关键字“dir”，由对应路径可找到模型中间产物的对应位置，方便调试。
