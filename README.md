本项目是基于 Attention U-Net 的一个医学图像分割项目。

## Directory
```
.
├── clear.py
├── Cursor.ipynb
├── data
│   ├── test
│   │   ├── 0.png
│   │   │ ......
│   │   ├── 29.png
│   └── train
│       ├── image
│       │   ├── 0.png
│       │   │  ......
│       │   ├── 29.png
│       └── label
│           ├── 0.png
│           │ ......
│           ├── 29.png
├── main.py
├── model
│   ├── AttentionBlock.py
│   ├── ConvBlock.py
│   ├── model.py
│   └── UpConvBlock.py
├── model_checkpoint
│   └── best_model.pth
├── parser.py
├── predict.py
├── train.py
├── ui
└── utils
    ├── dataset.py
    └── save_loss.py
```

## Usage:
`Cursor.ipynb` 用于在 UI 未被写出来之前模拟事件的触发。
我们可以在每个 Cell 中输入相应的指令:
`!python3 main.py --type="train"` / `!python3 main.py --type="predict"`
表示 Train / Predict 过程的开始。

在以上两条语句之前的 Cell 都不用管，用于在 Colab 中模拟 `shell` 访问文件目录(在 UI 写完之后当然就不用上啦)，但是如果需要在线允许，请根据自己的 Path 更改 `os` 相关语句。

## Data
```
├── data
│   ├── test
│   │   ├── 0.png
│   │   │ ......
│   │   ├── 29.png
│   └── train
│       ├── image
│       │   ├── 0.png
│       │   │  ......
│       │   ├── 29.png
│       └── label
│           ├── 0.png
│           │ ......
│           ├── 29.png
```
关于数据集，我们可以发现，仅有 30 条数据(train/test: 1: 1)。
在 `train` 中，原始图像及其Label一一对应，0 -> 29。
在 `test` 中，完成 `Predict` 过程的程序将会有输出结果 num\_\{i\}\_res.png(为什么现在没有，那是因为我 clear 掉了)，与测试图像对应。

## Others may confussing...
`parser.py`: 这是一个很有工程化的变量管理方式。通过 ArgsParameter or somehow 能够很方便的通过更改 args 的参数进行 fine tuning。

> 此外，我们上面的Usage中的 `!python3 main.py --type="xxx"` 的 `--type="xxx"` 可以通过 `shell` 将我们想要的参数传到 `main` 里，希望这对 UI 的构建有所帮助。

`save_loss`: For 逸非，画 loss 曲线。
 
## More Problem...
等待 Nick & Roy 撰写完成注释。
其实也能直接问我(Roy)。

## Reference
Thanks for [RABIA EDA YILMAZ](https://www.kaggle.com/code/truthisneverlinear/attention-u-net-pytorch)!!!