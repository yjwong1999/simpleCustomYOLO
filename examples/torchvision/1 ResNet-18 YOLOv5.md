## Example of ResNet18-based YOLOv5

Select a backbone from TorchVision
```python
from ultralytics.nn.modules.block import TorchVision
import torch

# load your prefered backbone
mods = TorchVision('resnet18', 'DEFAULT', True, 2, True)

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
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|██████████| 44.7M/44.7M [00:00<00:00, 125MB/s]
0 torch.Size([1, 3, 640, 640])
1 torch.Size([1, 64, 320, 320])
2 torch.Size([1, 64, 320, 320])
3 torch.Size([1, 64, 320, 320])
4 torch.Size([1, 64, 160, 160])
5 torch.Size([1, 64, 160, 160])
6 torch.Size([1, 128, 80, 80]) # we want this layer
7 torch.Size([1, 256, 40, 40]) # we want this layer
8 torch.Size([1, 512, 20, 20]) # we want this layer
```

From the outputs, you can see the outputs of the selected backbone. Usually, you can select the last 3 (or 4 depending on your needs) outputs as the intermediate features for the neck layer of your YOLO. For this example, we are modifying YOLOv5. Hence, we only need 3 intermediate features/outputs (layer 6 to 8) from the outputs above. Then, we add `SPPF` layers, similar to YOLOv5.
```yaml
backbone:
  # [from, number, module, args]
  - [-1, 1, TorchVision, [2048, "resnet18", "DEFAULT", True, 2, True]]  # - 0
  - [0, 1, Index, [128, 6]]   # selects 6th output (1, 512, 80, 80) - 1
  - [0, 1, Index, [256, 7]]  # selects 7th output (1, 1024, 40, 40) - 2
  - [0, 1, Index, [512, 8]]  # selects 8th output (1, 2048, 20, 20) - 3
  - [-1, 1, SPPF, [1024, 5]] # SPFF - 4
```
☝️ Notice that we use `Index` layer to select which outputs from the `TorchVision` layer. The `TorchVision` layer has 9 outpus (0 to 9) as shown in the backbone layout above. We need layers 6 to 8. Hence, we use `[0, 1, Index, [128, 6]]`, `[0, 1, Index, [256, 7]]` and `[0, 1, Index, [512, 8]]` to select the layers. Note that the `128, 256, 512` are the number of channels for each of these selected layers, while `6, 7, 8` are the selected outputs index from the `ResNet18`. Please modify according to your selected backbone/layers.

For the neck and detection heads, you can just copy from the YOLO version you want. For example, we can copy from YOLOv5, as shown below:
```yaml
head:
  # [from, number, module, args]
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
☝️ Note that you would need to modify the arguments `from` to make sure the layers are connected to your custom backbone at the correct layer. The example above has been modified to fit YOLOv5 neck and head with our custom `ResNet18`.

Finally, the entire `yaml` file for ResNet18-YOLOv5 can be written as follows:
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
