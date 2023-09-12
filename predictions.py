import os

import configargparse
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from scipy.spatial import distance as dist
from torchvision import models, transforms


def initParser():
    parser = configargparse.get_argument_parser()
    parser.add_argument("images", nargs="+", help="list of images")
    parser.add_argument("--step", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False, help="save yolo dataset")
    parser.add_argument("--model", default="./model_v2.pth")
    return parser.parse_args()


def create_model(weights="model_v2.pth", outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, outputchannels)
    checkpoint = torch.load(weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    _ = model.eval()

    if torch.cuda.is_available():
        model.to("cuda")
    return model


def pred(image, model, threshold=0.1):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")

    with torch.no_grad():
        output = model(input_batch)["out"][0]
        output = (output > threshold).type(torch.IntTensor)
        output = output.cpu().numpy()[0]

        return output


def segmentate(frame, model):
    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    outputs = pred(image=image, model=model)
    result = np.where(outputs > 0)
    coords = list(zip(result[0], result[1]))
    for coord in coords:
        image.putpixel((coord[1], coord[0]), (0, 255, 0))

    out = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    return out, coords


def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def getBoxes(frame, coords):
    h, w = frame.shape[:2]

    # create black image and white pixels on segmentation
    img = np.zeros((h, w, 1), dtype=np.uint8)
    for coord in coords:
        x, y = coord
        img[x, y] = 255

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        #peri = cv.arcLength(cnt, True)
        #box = np.int0([i[0] for i in cv.approxPolyDP(cnt, 0.05 * peri, True)])
        
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        if len(box) == 4:
            boxes.append(box)

    return boxes


def getWarped(frame, box, w=300, h=100):
    rect = order_points_new(box)
    destpts = np.float32([[0, 0], [300, 0], [300, 100], [0, 100]])
    resmatrix = cv.getPerspectiveTransform(np.float32(rect), destpts)
    return np.int0(rect), cv.warpPerspective(frame, resmatrix, (300, 100))


def overlayImage(frame, xy, plate):
    x_offset, y_offset = xy
    frame[
        y_offset : y_offset + plate.shape[0], x_offset : x_offset + plate.shape[1]
    ] = plate

def equalize(frame):
    yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
    yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0]) # equalize the histogram of the Y channel
    return cv.cvtColor(yuv, cv.COLOR_YUV2BGR)


def saveDataset(frame, filename, boxes):
    h,w = frame.shape[:2]
    with open(filename, "w") as f:
        for box in boxes:
            f.write("0")
            for coord in box:
                f.write(" %s %s" % (float(coord[0])/w, float(coord[1])/h))
            f.write("\n")


def main():
    config = initParser()
    model = create_model(config.model)
    for filename in config.images:
        frame = cv.imread(filename)
        if len(frame.shape) == 2:
            frame = cv.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        image, coords = segmentate(frame, model)

        boxes = getBoxes(frame, coords)

        if config.save:
            saveDataset(frame, os.path.splitext(filename)[0] + ".txt", boxes)

        for box in boxes:
            rect, plate = getWarped(frame, box)
            cv.drawContours(frame, [box], 0, (0, 0, 255), 1)

            plate = equalize(plate)

            plate = cv.copyMakeBorder(src=plate, top=2, bottom=2, left=2, right=2, borderType=cv.BORDER_CONSTANT, value=(255,0,0)) 

            try:
                y = sorted(rect, key=lambda p: p[1])[0][
                    1
                ]  # prendiamo il punto più alto
                x = sorted(rect, key=lambda p: p[0])[0][
                    0
                ]  # prendiamo il punto più a sinistra
                overlayImage(frame, (x, max(1, y - plate.shape[0])), plate)
            except Exception as e:
                print(e)
                continue
        cv.imshow("image", np.concatenate((image, frame), axis=1))
        key = cv.waitKey(0 if config.step else 1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
