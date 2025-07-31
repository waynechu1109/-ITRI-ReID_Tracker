<h2 align="center">
  <strong>
    <a href="README.md">English</a> | <a href="README_ch.md">中文</a>
  </strong>
</h2>

# 人物 Re-ID 辨識

## Docker 環境建制
```bash
sudo docker run -it --net=host --gpus=all --rm=false  --name tao-toolkit-5.5.0-deploy \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v $HOME/ros2_ws/:/root/ros2_ws/ \
  -v /etc/timezone:/etc/timezone \
  -v /etc/localtime:/etc/localtime \
  nvcr.io/nvidia/tao/tao-toolkit:5.5.0-deploy
```

## 將 `.pt` 轉為 `.onnx`
```bash
$ cd ros2_ws/deep-person-reid/tools

# 執行轉換
$ python3 export.py \
    --dynamic \
    --include onnx \
    --weights /root/ros2_ws/src/reid_tracker/models/osnet_ain_x1_0_ms_d_c.pt
```

## 將 `.onnx` 轉為 `.engine`
先將上一步驟所產生的 `.onnx` 檔移動至 `src/reid_tracker/models` 之下。
```bash
$ cd ros2_ws/src/reid_tracker/models

# 執行轉換
$ trtexec \
    --onnx=osnet_ain_x1_0_ms_d_c.onnx \
    --minShapes=images:1x3x256x128 \
    --optShapes=images:64x3x256x128 \
    --maxShapes=images:128x3x256x128 \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --saveEngine=osnet_ain_x1_0_ms_d_c.engine \
    --verbose > trt_build.log

# --minShapes 表示 engine 最小可能處裡的維度 (batchsize = 1)
# --optShapes 表示 engine 最常可能處裡的維度，並針對此維度進行優化 (batchsize = 64)
# --maxShapes 表示 engine 最大可能處裡的維度 (batchsize = 128)

# 可使用以下指令來測試 engine 效能
trtexec \
  --loadEngine=osnet_ain_x1_0_ms_d_c.engine \
  --shapes=images:60x3x256x128 \
  --warmUp=10 \
  --iterations=1000 \
  --avgRuns=100

# 執行 10 次推論以預熱 GPU、避免冷啟導致結果偏差
# 執行 1000 次推論來進行效能測試
# 取 100 次的平均，提升結果準確性
```
最後即可在 `models` 資料夾中看到轉換成功的 `.engine`。


## 執行 ReID 範例
```bash
# 需同步播放 ros2bag 來提供輸入畫面
$ cd ros2_ws/rosbag2/

# 播放
$ ros2 bag play street_resized_resized/ -l

```
先使用 `ros2 topic list` 確認是否能看到 `/camera/image_raw` 這個 topic。

```bash
$ cd ros2_ws

# build
$ colcon build --packages-select reid_tracker
$ source install/setup.bash

# 執行主程式
$ ros2 launch reid_tracker person_reid_batch.launch.py
```
開始追蹤畫面中特定人物，只需要使用滑鼠點擊，隨後該人物偵測匡就會變成紅色，程式便可以實現追蹤。如要切換追蹤人物，直接點擊其他人即可。

輸出的偵測畫面會被發布在 `/person_reid/image` 這個 topic，偵測人物框則會被發布在 `/person_reid/bboxes`。