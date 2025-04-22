# simpleCustomYOLO

This is a GitHub repo to teach you how to easily change your `ultralytics` yolo backkbone with pretrained backbone from `TorchVision` and also `Timm`, without introducing extra codes manually. All you need is learn to configure the `yaml` file. That's it!

📋 Useful Links:
1. [TorchVision examples](https://github.com/yjwong1999/simpleCustomYOLO/tree/main/examples/torchvision)
2. [Timm examples, pending](https://github.com/yjwong1999/simpleCustomYOLO/tree/main/examples/torchvision)

Here is a simple examples of ResNet-18-based YOLOv5. No extra codes needed! Only `yaml` file required.

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license

# Parameters
nc: 80 # number of classes

backbone:
  # [from, number, module, args]
  - [-1, 1, TorchVision, [2048, "resnet18", "DEFAULT", True, 2, True]]  # - 0
  - [0, 1, Index, [128, 6]]   # selects 6th output (1, 512, 80, 80) - 1
  - [0, 1, Index, [256, 7]]  # selects 7th output (1, 1024, 40, 40) - 2
  - [0, 1, Index, [512, 8]]  # selects 8th output (1, 2048, 20, 20) - 3
  - [-1, 1, SPPF, [1024, 5]] # SPFF - 4

head:
  - [-1, 1, Conv, [512, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C3, [512, False]] # 8

  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C3, [256, False]] # 12 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]] # cat head P4
  - [-1, 3, C3, [512, False]] # 15 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 4], 1, Concat, [1]] # cat head P5
  - [-1, 3, C3, [1024, False]] # 18 (P5/32-large)

  - [[12, 15, 18], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

To train your model, save the yaml file as lets say `yolov5-resnet18.yaml`. Then, just load the model using `YOLO` and train as usual.

```python
from ultralytics import YOLO

# Load your custom model
model = YOLO("yolov5-resnet18.yaml")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/path/to/your/data.yaml", epochs=20, imgsz=960, batch=16, val=False)
```

📢Tutorials for YOLO customization using Timm will be out soon!
