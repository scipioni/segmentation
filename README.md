# segmentation


requirements
```
yay -S python-torchvision-cuda
task venv
```

download model_v2.pth from https://github.com/dbpprt/pytorch-licenseplate-segmentation


autolabel jpg into txt (yolo segmentation)
```
python predictions.py /archive/dataset/plates/train/*jpg
```

convert yolo segmentation to labelstudio /tmp/project.json
```
cp -a /archive/dataset/plates/train/ /tmp/plates
python yolo2ls.py /tmp/plates/*.txt
```