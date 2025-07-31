
<!-- <h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.png" alt="Markdownify" width="200"></a>
  <br>
  Markdownify
  <br>
</h1>

<h4 align="center">A minimal Markdown Editor desktop app built on top of <a href="http://electron.atom.io" target="_blank">Electron</a>.</h4>

<p align="center">
  <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://badge.fury.io/js/electron-markdownify.svg"
         alt="Gitter">
  </a>
  <a href="https://gitter.im/amitmerchant1990/electron-markdownify"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
  <a href="https://saythanks.io/to/bullredeyes@gmail.com">
      <img src="https://img.shields.io/badge/SayThanks.io-%E2%98%BC-1EAEDB.svg">
  </a>
  <a href="https://www.paypal.me/AmitMerchant">
    <img src="https://img.shields.io/badge/$-donate-ff69b4.svg?maxAge=2592000&amp;style=flat">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

![screenshot](https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/app/img/markdownify.gif)

## Key Features

* LivePreview - Make changes, See changes
  - Instantly see what your Markdown documents look like in HTML as you create them.
* Sync Scrolling
  - While you type, LivePreview will automatically scroll to the current location you're editing.
* GitHub Flavored Markdown  
* Syntax highlighting
* [KaTeX](https://khan.github.io/KaTeX/) Support
* Dark/Light mode
* Toolbar for basic Markdown formatting
* Supports multiple cursors
* Save the Markdown preview as PDF
* Emoji support in preview :tada:
* App will keep alive in tray for quick usage
* Full screen mode
  - Write distraction free.
* Cross platform
  - Windows, macOS and Linux ready. -->

# OSNet 加速

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


<!-- ## Emailware

Markdownify is an [emailware](https://en.wiktionary.org/wiki/emailware). Meaning, if you liked using this app or it has helped you in any way, I'd like you send me an email at <bullredeyes@gmail.com> about anything you'd want to say about this software. I'd really appreciate it!

## Credits

This software uses the following open source packages:

- [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/)

## Related

[Try Web version of Markdownify](https://notepad.js.org/markdown-editor/)

## Support

If you like this project and think it has helped in any way, consider buying me a coffee!

<a href="https://buymeacoffee.com/amitmerchant" target="_blank"><img src="app/img/bmc-button.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

## You may also like...

- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS

## License

MIT

---

> [amitmerchant.com](https://www.amitmerchant.com) &nbsp;&middot;&nbsp;
> GitHub [@amitmerchant1990](https://github.com/amitmerchant1990) &nbsp;&middot;&nbsp;
> Twitter [@amit_merchant](https://twitter.com/amit_merchant)
 -->
