#!/bin/bash

# Step 1. 将所有18k个图像和json文件copy到对应目录下
# copy all images into detseg/dataset/all_data/images
cp path/to/data/初赛/train/image/* ./detseg/dataset/all_data/images/
cp path/to/data/初赛/补充1000张训练/image/* ./detseg/dataset/all_data/images/
cp path/to/data/决赛/给选手的图片/* ./detseg/dataset/all_data/images/
# copy all json files into detseg/dataset/all_data/label
cp path/to/data/初赛/train/label/* ./detseg/dataset/all_data/label/
cp path/to/data/初赛/补充1000张训练/label/* ./detseg/dataset/all_data/label/
cp path/to/data/决赛/给选手的标签/* ./detseg/dataset/all_data/label/


# Step 2. 解析json文件，会产生train_det.json、val_det.json和val_seg.json三个文件
#         并且在./label_mask目录下产生18k个.npy文件 新建了指向all_data/images的软链images.
cd ./detseg/dataset
python parse_train_val.py
ln -s ./all_data/images images
cd ..

# Step 3. 当前目录在detseg/，开始训练
# 推荐使用nohup后台执行，以免网络不稳定导致中断。
nohup python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --eval > log.txt &

# ************************** 关于训练中断的异常处理***********************************
# 如果训练过程中有意外中断，可以执行下面的命令继续训练，其中最后的数字为训练终止的最后一个epoch，这里设置的是每间隔4个epoch做一次评估和保存，所以最后一个epoch会是3、7、11、15、19、23……这样的位置。
# 比如训练在22个epoch中断，则应当用下面的命令来继续训练
# 如果没有中断，则只需要等待就可以了。
# python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --eval --resume output/mot_ppyoloe_m_36e_xd_v8/19
# 中断时，可能会产生best_model实际上并不是整个训练过程中最佳的模型，而是从继续训练点之后最佳的模型。比如，在epoch=83时中断，之前的最佳结果为0.425；从83继续训练时，best_model存储的只是83之后的最佳结果，不排除该最佳结果低于0.425的可能。此时，要将真正的最佳模型命名为best_model.pdparams。
# mv ./output/mot_ppyoloe_m_36e_xd_v8/43.pdparams ./output/mot_ppyoloe_m_36e_xd_v8/best_model.pdparams
# 此处，容易产生歧义，最好的状态就是不产生中断：）
# *************************************************************************************

# Step 4. 测试并判定阈值
python tools/eval.py --config configs/exp/mot_ppyoloe_m_36e_xd_v8.yml
# 此时，./detseg目录下，应当会产生bbox.json和seg.json两个文件，依次执行下面两条命令
python select_F1_thresh.py
python select_F1_seg_thresh.py
# 至此，./detseg目录下，应当产生det_threshes.txt文件，该文件中包含11个阈值。

# Step 5. 导出模型
python tools/export_model.py -c configs/exp/mot_ppyoloe_m_36e_xd_v8.yml --output_dir=./inference_model/ -o weights=output/mot_ppyoloe_m_36e_xd_v8/best_model.pdparams
mv ./inference_model/mot_ppyoloe_m_36e_xd_v8 ../deploy/model/v8
mv ./det_threshes.txt ../deploy/model/v8/

# 至此，所有步骤都已经执行完毕，deploy就是我们要压缩并提交的文件夹.
# test_images.txt存储的是待测试图像的地址列表.
cd ../deploy
python predict.py test_images.txt result.json
