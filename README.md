# 基于视听信息的音源分离与定位
2018-2019 学年度秋季学期《视听信息系统导论》课程设计

## 基本思路
利用 Pytorch 提供的预训练好的 densenet161 实现音源定位和乐器判断，训练 Unet 实现音源分离

## 代码运行方法
下述的 py 文件都可以运行 `python a.py -h` 查看具体参数含义

### 训练

```bash
python generate_dataset_list.py --audio_root /path/to/audio/root
python train.py
```

### 测试

完整 Pipeline：

```bash
# 处理视频，给出每个视频对应的乐器种类和位置
python generate_test_info_from_video.py --video_dir /path/to/video/root --audio_dir /path/to/audio/root
# 运行音源分离和评估
bash run_test.sh /path/to/model.pth False /name/of/result.txt
```

分步：

```bash
# 处理视频，给出每个视频对应的乐器种类和位置 json 文件
python generate_test_info_from_video.py --video_dir /path/to/video/root --audio_dir /path/to/audio/root --output /path/to/result/of/video.json
# 音源分离
python test.py --pretrained_model /path/to/model.pth --file_list /path/to/result/of/video.json
# 评估（该函数不能查看参数定义）
python Evaluate.py /name/of/result.txt
```


