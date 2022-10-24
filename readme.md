# A Dual-task and end-to-end Network for Detection and Semantic Segmentation

# Overview
本项目旨在检测图像中的车辆、非机动车、行人、红灯、绿灯、黄灯、未亮灯7类目标，同时对车道线中实线、虚线与人行道线3类进行语义分割。我们设计了一个将目标检测与语义分割融合的dual-task end2end网络，从而实现在单个网络解决本场景中的问题。同时使得检测与分割从相互的内容信息中得到辅助，从而都得到了提高，尤其检测对分割结果的提高作用更加明显。
其次，对于不均衡和缺失的类别，我们设计了一套更加贴近现实的基于instaboost的增强方式，替代原有的copy-paste策略。我们利用街景图像的语义分割结果估计目标出现的可能位置，而不是随机进行位置选择，从而使得目标出现位置更加合理。
最后，针对小目标检测和模型速度优化的问题，我们在architecture对比、backbone、input resolution、large feature map、多尺度训练、训练参数、阈值确定、多进程并行等多个方面进行对比实验，大大提高了模型的准确率和速度，达到最优的平衡。
效果展示：
<video id="video" controls="" preload="none" poster="">
      <source id="mp4" src="https://www.bilibili.com/video/BV1og41187E8" type="video/mp4">
      <p>Your user agent does not support the HTML5 Video element.</p>
</video>

# 模型训练
## Step 1. 
将所有18k个图像和json文件copy到对应目录下
```
cp path/to/data/初赛/train/image/* ./detseg/dataset/all_data/images/
cp path/to/data/初赛/补充1000张训练/image/* ./detseg/dataset/all_data/images/
cp path/to/data/决赛/给选手的图片/* ./detseg/dataset/all_data/images/
cp path/to/data/初赛/train/label/* ./detseg/dataset/all_data/label/
cp path/to/data/初赛/补充1000张训练/label/* ./detseg/dataset/all_data/label/
cp path/to/data/决赛/给选手的标签/* ./detseg/dataset/all_data/label/
```

## Step 2. 
解析json文件，会产生train_det.json、val_det.json和val_seg.json三个文件
并且在./label_mask目录下产生18k个.npy文件 新建了指向all_data/images的软链images.
```
cd ./detseg/dataset
python parse_train_val.py
ln -s ./all_data/images images
cd ..
```

## Step 3. 
当前目录在detseg/，开始训练
推荐使用nohup后台执行，以免网络不稳定导致中断。
```
nohup python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --eval > log.txt &
```

* 关于训练中断的异常处理 *
如果训练过程中有意外中断，可以执行下面的命令继续训练，其中最后的数字为训练终止的最后一个epoch，这里设置的是每间隔4个epoch做一次评估和保存，所以最后一个epoch会是3、7、11、15、19、23……这样的位置。
比如训练在22个epoch中断，则应当用下面的命令来继续训练
如果没有中断，则只需要等待就可以了。
```
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --eval --resume output/mot_ppyoloe_m_36e_xd_v8/19
```
中断时，可能会产生best_model实际上并不是整个训练过程中最佳的模型，而是从继续训练点之后最佳的模型。比如，在epoch=83时中断，之前的最佳结果为0.425；从83继续训练时，best_model存储的只是83之后的最佳结果，不排除该最佳结果低于0.425的可能。此时，要将真正的最佳模型命名为best_model.pdparams。
```
mv ./output/mot_ppyoloe_m_36e_xd_v8/43.pdparams ./output/mot_ppyoloe_m_36e_xd_v8/best_model.pdparams
```
此处，容易产生歧义，最好的状态就是不产生中断：）

## Step 4. 
测试并判定阈值
```
python tools/eval.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml
```
此时，./detseg目录下，应当会产生bbox.json和seg.json两个文件，依次执行下面两条命令
```
python select_F1_thresh.py
python select_F1_seg_thresh.py
```
至此，./detseg目录下，应当产生det_threshes.txt文件，该文件中包含11个阈值。

## Step 5. 
导出模型
```
python tools/export_model.py -c configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --output_dir=./inference_model/ -o weights=output/mot_ppyoloe_m_36e_xd_v8/best_model.pdparams
mv ./inference_model/mot_ppyoloe_m_36e_xd_v8 ../deploy/model/v8
mv ./det_threshes.txt ../deploy/model/v8/
```

至此，所有步骤都已经执行完毕，deploy就是我们要压缩并提交的文件夹.
test_images.txt存储的是待测试图像的地址列表.
```
cd ../deploy
python predict.py test_images.txt result.json
```
