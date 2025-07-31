# [ITRI Internship] Person Re-ID Tracker
<!-- 
This project is largely based on [ros2_tao_pointpillars](https://github.com/NVIDIA-AI-IOT/ros2_tao_pointpillars). The code is modified for Mask2Former TensorRT .engine file inference. -->


## Quick Start
### Docker
```bash
sudo docker run -it --net=host --gpus=all --rm=false  --name tao-toolkit-5.5.0-deploy \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v $HOME/ros2_ws/:/root/ros2_ws/ \
  -v /etc/timezone:/etc/timezone \
  -v /etc/localtime:/etc/localtime \
  nvcr.io/nvidia/tao/tao-toolkit:5.5.0-deploy
```

### Accelerate OSNet
OSNet is used for 512D features extraction in this project. OSNet model can be downloaded [here](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO).

We have to first convert the `.pt` OSNet model file to `.onnx` file for further speed up. 

```bash
cd tools/
# run conversion
python3 export.py \
    --dynamic \
    --include onnx \
    --weights /root/ros2_ws/src/reid_tracker/models/osnet_ain_x1_0_ms_d_c.pt
```

After that, we can build OSNet TensorRT `.engine` for full acceleration. 

Move the `.onnx` file generated last step to `src/reid_tracker/models` folder. 
```bash
cd ros2_ws/src/reid_tracker/models
# run tensorrt .engine generation
trtexec \
    --onnx=osnet_ain_x1_0_ms_d_c.onnx \
    --minShapes=images:1x3x256x128 \
    --optShapes=images:64x3x256x128 \
    --maxShapes=images:128x3x256x128 \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw \
    --saveEngine=osnet_ain_x1_0_ms_d_c.engine \
    --verbose > trt_build.log
```

- `--minShapes` indicates the minimum dimension that the `.engine` file could process (batchsize = 1).
- `--optShapes` specifies the most commonly expected input dimensions, allowing the engine to be optimized for those shapes (batchsize = 64).
- `--maxShapes` indicates the maximum dimension that the `.engine` file could process (batchsize = 128).

You can also test the exported `.engine` with the following command.

```bash
trtexec \
  --loadEngine=osnet_ain_x1_0_ms_d_c.engine \
  --shapes=images:60x3x256x128 \
  --warmUp=10 \
  --iterations=1000 \
  --avgRuns=100
```

- `--warmUp`: Perform 10 inference runs to warm up the GPU and avoid cold-start bias.
- `--iterations=1000`: Run 1000 inferences for performance benchmarking.
- `avgRuns=100`: Take the average over 100 runs to improve the accuracy of the results.

### Run ReID
In the prototype stage, you can play ROS2 bag 

```bash
# 需同步播放 ros2bag 來提供輸入畫面
cd ros2_ws/rosbag2/

# 播放
ros2 bag play street_resized_resized/ -l

```
先使用 `ros2 topic list` 確認是否能看到 `/camera/image_raw` 這個 topic。

```bash
cd ros2_ws

# build
colcon build --packages-select reid_tracker
source install/setup.bash

# 執行主程式
ros2 launch reid_tracker person_reid_batch.launch.py
```
開始追蹤畫面中特定人物，只需要使用滑鼠點擊，隨後該人物偵測匡就會變成紅色，程式便可以實現追蹤。如要切換追蹤人物，直接點擊其他人即可。

輸出的偵測畫面會被發布在 `/person_reid/image` 這個 topic，偵測人物框則會被發布在 `/person_reid/bboxes`。



## Results
<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 100%; vertical-align: top;">
      <div style="width: 100%; text-align: center;">
        <img src="readme_media/reid_tracker_shorten.gif" style="width: 100%;" />
        <!-- <div style="margin-top: 8px;">Mapillary</div> -->
      </div>
    </td>
  </tr>
</table>


## Related Porjects

## Acknowledgement
