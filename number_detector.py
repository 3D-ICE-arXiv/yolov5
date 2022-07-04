import argparse
import os
import sys
from pathlib import Path
from tkinter.tix import Tree
import numpy as np
import torch
import cv2
import torch.backends.cudnn as cudnn
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class NumberDetectionYolo:
    def __init__(self, weight, yaml_data):
        # Load model
        self.device=0 # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.dnn=False # use OpenCV DNN for ONNX inference
        self.half=False # use FP16 half-precision inference
        self.device = select_device(self.device)
        self.weights = weight #r"C:\Users\lordo\PycharmProjects\ShotChart\yolov5\runs\train\exp4\weights\best.pt"
        self.yaml_data = yaml_data #r"C:\Users\lordo\PycharmProjects\ShotChart\yolov5\runs\train\exp4\data.yaml"
        self.imgsz=(640, 640)  # inference size (height, width)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.yaml_data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        #imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    def file2cvimg(self, file_path):

        #img_path = r"D:\67_Basketball\23_numberdetection\PlayerClassificationDataset\train\color\960_262.png"
        img0 = cv2.imread(file_path)
        return img0

    def file2tensor(self, file_path):
        cv2_img = self.file2cvimg(file_path)
        return self.cv2tensor(cv2_img)

    def cv2tensor(self, cv2_img):
        img_size=640
        img = letterbox(cv2_img, img_size, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        #print("imshape"+str(im.shape))
        im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
        #print("imshape"+str(im.shape))
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def get_backnumber_from_cv2(self, cv2_img):
        #get output from cv2 file
        img_tensor = self.cv2tensor(cv2_img)
        return self.get_backnumber(img_tensor)

    def get_backnumber(self, img_tensor):
        # calculate YOLO output. 
        # return value is x(upper left), y(upper left), x(bottom right), y(bottom right), possiblity, number
        # return value can be array if the image has more than 2 output.
        
        # Inference
        augment=False
        visualize=False
        pred = self.model(img_tensor, augment=augment, visualize=visualize)

        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        max_det=1000  # maximum detections per image
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred_numpy = pred[0].to('cpu').detach().numpy().copy()
        return pred_numpy, len(pred_numpy)

    def get_backnumber_from_path(self, file_path):
        #get output from image file
        tensor_img = self.file2tensor(file_path)
        return self.get_backnumber(tensor_img)

if __name__ == '__main__':

    weights = r"C:\Users\lordo\PycharmProjects\ShotChart\yolov5\runs\train\exp4\weights\best.pt"
    yaml_data = r"C:\Users\lordo\PycharmProjects\ShotChart\yolov5\runs\train\exp4\data.yaml"
    yolov5 = NumberDetectionYolo(weights, yaml_data)

    file_path2 = r"D:\67_Basketball\23_numberdetection\PlayerClassificationDataset\train\color\960_262.png"
    file_path1 = r"D:\67_Basketball\23_numberdetection\PlayerClassificationDataset\train\color\360_116.png"
    file_path0 = r"D:\67_Basketball\23_numberdetection\PlayerClassificationDataset\train\color\330_6.png"
    cv2img = cv2.imread(file_path2)
    print("cv2")
    pred , num= yolov5.get_backnumber_from_cv2(cv2img)
    for i in range(0,num):
        print(pred[i])
    '''
    pred , num= yolov5.file2get_backnumber(file_path0)
    for i in range(0,num):
        print(pred[i])

    pred , num= yolov5.file2get_backnumber(file_path1)
    for i in range(0,num):
        print(pred[i])
        for item in pred[i]:
            print(item)
    '''
    print("from file")
    pred , num= yolov5.file2get_backnumber(file_path2)
    for i in range(0,num):
        print(pred[i])
        print(pred[i][0])
        print(pred[i][1])
        print(pred[i][2])
        print(pred[i][3])
        print(pred[i][4])
        print(pred[i][5])

