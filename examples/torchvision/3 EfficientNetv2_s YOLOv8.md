## Example of EfficientNetv2_s-based YOLOv8

Select a backbone from TorchVision
```python
from ultralytics.nn.modules.block import TorchVision
import torch

# load your prefered backbone
mods = TorchVision('efficientnet_v2_s', 'DEFAULT', True, 2, True)

# print each module in mods
# mods.m

# init a random tensor as a dummy image with your desired image size
img = torch.randn(1, 3, 640, 640)

# inference and print out the shape
out = mods(img)
for i in range(len(out)):
    print(i, out[i].shape)
```

Sample outputs:
```
0 torch.Size([1, 3, 640, 640])
1 torch.Size([1, 24, 320, 320])
2 torch.Size([1, 24, 320, 320])
3 torch.Size([1, 48, 160, 160])
4 torch.Size([1, 64, 80, 80])   # we want this layer
5 torch.Size([1, 128, 40, 40]) 
6 torch.Size([1, 160, 40, 40])  # we want this layer
7 torch.Size([1, 256, 20, 20])
8 torch.Size([1, 1280, 20, 20]) # we want this layer
```

👆 From the outputs, you can see the outputs of the selected backbone. As mentioned in the first 2 tutorials, the outputs are useful for us to construct the `yaml` file.

⚠️ If you recall back from our tutorial of ResNet18-based YOLOv5/v8, we extracted the last 3 outputs of torchvision ResNet18 as the intermediate features for neck layer. 

However, for this backbone (EfficientNetv2_s), we cannot select the last 3 outputs. The reason is simple, we need to make sure the 3 intermediate outputs have resolution size of X2 from the smallest to largest. 

If we take the last 3 layers (6, 7, 8), the resolutions of the features are `40x40`, `20x20`, and `20x20`, respectively. However, this does not match the requirements earlier.

Instead, we first find out the last layers with resolulions `80x80`, `40x40`, and `20x20` first. This happens to be layers `4`, `6`, and `8`, respectively.

The rest are similar to our previous tutorials. We use `Index` to select layers `4`, `6`, and `8`. Then, attach `SPPF` layer, similar to YOLOv8. (If you are using other YOLO, please check if they use SPPF or other layers).

With that, the backbone of EfficientNetv2_s-based YOLOv8 can be expressed as follows:
```yaml
backbone:
  # [from, number, module, args]
  - [-1, 1, TorchVision, [2048, "efficientnet_v2_s", "DEFAULT", True, 2, True]]  # - 0
  - [0, 1, Index, [64, 4]]   # selects 4th output (1, 64, 80, 80) - 1
  - [0, 1, Index, [160, 6]]  # selects 6th output (1, 160, 40, 40) - 2
  - [0, 1, Index, [1280, 8]]  # selects 8th output (1, 1280, 20, 20) - 3
  - [-1, 1, SPPF, [1024, 5]] # SPFF - 4
```
☝️ Notice that we use `Index` layer to select which outputs from the `TorchVision` layer. The `TorchVision` layer has 9 outputs (0 to 8) as shown in the backbone layout above. We need layers 4, 6, 8. Hence, we use `[0, 1, Index, [64, 4]]`, `[0, 1, Index, [160, 6]]` and `[0, 1, Index, [1280, 8]]` to select the layers. Note that the `64, 160, 1280` are the number of channels for each of these selected layers, while `4, 6, 8` are the selected outputs index from the `efficientnet_v2_s`. Please modify according to your selected backbone/layers.

For the neck and detection heads, you can just copy from the YOLO version you want. For example, we can copy from YOLOv8, as shown below:
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 7

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 10 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 7], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 13 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 4], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 16 (P5/32-large)

  - [[10, 13, 16], 1, Detect, [nc]] # Detect(P3, P4, P5)
```
☝️ Note that you would need to modify the arguments `from` to make sure the layers are connected to your custom backbone at the correct layer. The example above has been modified to fit YOLOv8 neck and head with our custom `EfficientNetv2_s`.

Finally, the entire `yaml` file for ResNet18-YOLOv8 can be written as follows:
```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Ultralytics YOLO 🚀, AGPL-3.0 license

# Parameters
nc: 80 # number of classes

backbone:
  # [from, number, module, args]
  - [-1, 1, TorchVision, [2048, "efficientnet_v2_s", "DEFAULT", True, 2, True]]  # - 0
  - [0, 1, Index, [64, 4]]   # selects 4th output (1, 64, 80, 80) - 1
  - [0, 1, Index, [160, 6]]  # selects 6th output (1, 160, 40, 40) - 2
  - [0, 1, Index, [1280, 8]]  # selects 8th output (1, 1280, 20, 20) - 3
  - [-1, 1, SPPF, [1024, 5]] # SPFF - 4

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 7

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 10 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 7], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 13 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 4], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 16 (P5/32-large)

  - [[10, 13, 16], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

To train your model, save the yaml file as lets say `yolov8-efficientnetv2_S.yaml`. Then, just load the model using `YOLO` and train as usual.

```python
from ultralytics import YOLO

# Load your custom model
model = YOLO("yolov8-efficientnetv2_S.yaml")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/path/to/your/data.yaml", epochs=20, imgsz=960, batch=16, val=False)
```

⚠️ Note that while EfficientNet family are designed to be lightweight and fast, they do not run fast in GPU. This is because the architecture used by EfficientNet and MobileNet are generally more optimized for CPU. Hence, you will notice that even though EfficientNetv2_s has smaller parameters, it cannot be run with large batch size compared to other models like ResNet.
