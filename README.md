# segmentation


requirements
```
yay -S python-torchvision-cuda
task venv
```

download model_v2.pth from https://github.com/dbpprt/pytorch-licenseplate-segmentation


test
```
python predictions.py /archive/dataset/plates/train/*jpg
```