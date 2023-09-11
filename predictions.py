#!/usr/bin/env python
# coding: utf-8

# yay -S python-torchvision-cuda
# pip install configargparse
# scaricare il modello model_v2.pth da https://github.com/dbpprt/pytorch-licenseplate-segmentation

import torch
import cv2
from PIL import Image
from torchvision import transforms
from torchvision import models
import numpy as np
import configargparse


def initParser():
    parser = configargparse.get_argument_parser()
    parser.add_argument("images", nargs="+", help="list of images")
    return parser.parse_args()

def create_model(weights="model_v2.pth", outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, outputchannels)
    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    if torch.cuda.is_available():
        model.to('cuda')
    return model

def pred(image, model, threshold=0.1):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')

    with torch.no_grad():
        print("inference")
        output = model(input_batch)['out'][0]
        print("ok")
        output = (output > threshold).type(torch.IntTensor)
        output = output.cpu().numpy()[0]

        return output

def segmentate(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    outputs = pred(image=image, model=model)
    result = np.where(outputs > 0)
    coords = list(zip(result[0], result[1]))
    for cord in coords:
        image.putpixel((cord[1], cord[0]), (255, 0, 0))

    out = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return out, coords 


def main():
    config = initParser()
    model = create_model("./model_v2.pth")
    for filename in config.images:
        frame = cv2.imread(filename)

        image,coords = segmentate(frame, model)
        
        cv2.imshow("image", image)
        key=cv2.waitKey(0)
        if key==ord("q"):
            break

if __name__=="__main__":
    main()