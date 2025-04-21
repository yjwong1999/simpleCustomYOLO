# simpleCustomYOLO

## Example of ResNet18-based YOLOv5

Select a backbone from TorchVision
```python
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
6 torch.Size([1, 128, 80, 80])
7 torch.Size([1, 256, 40, 40])
8 torch.Size([1, 512, 20, 20])
```

From the outputs, you can see the outputs of the selected backbone. Usually, you can select the last 3 (or 4 depending on your needs) outputs as the intermediate features for the neck layer of your YOLO. 
