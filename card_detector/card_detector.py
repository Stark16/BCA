import os
import sys
import numpy as np
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
# from utils.dataloaders import (IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams)
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device


class CardInference:

    def __init__(self, PATH_model:str, img_size:tuple=(512, 512), conf_thresh:float=0.25, iou_thresh:float=0.35, device:str='', hide_labels:bool=False, hide_conf:bool=False, view_img:bool=True, save_img:bool=True) -> None:
        self.PATH_weights = PATH_model
        self.IMG_SIZE = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.view_img = view_img
        self.save_img = save_img

        # Load Model:
        print('[INFO] Loading Model ...')
        device = select_device(device)
        self.model = DetectMultiBackend(self.PATH_weights, device=device, dnn=False, data=None, fp16=False)
        self.STRIDE, self.names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.IMG_SIZE, s=self.STRIDE)  # check image size
        self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *imgsz))  # warmup

    def image_preproc(self, image:np.ndarray):
        im = letterbox(image, self.IMG_SIZE, stride=self.STRIDE, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        return im

    def process(self, input_img:np.ndarray):

        im = self.image_preproc(input_img)
        seen, _, dt = 0, [], (Profile(), Profile(), Profile())
        # for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = self.model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, max_det=1000)

        # Process predictions
        predictions = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = input_img.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # if save_img or save_crop or view_img:  # Add bbox to image
                    # c = int(cls)  # integer class
                    # label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    if (self.view_img):
                        cv2.rectangle(im0, p1, p2, (56, 56, 255), 3)
                    predictions.append([p1, p2], conf)

            if self.view_img:
                cv2.imshow(str('img'), im0)
                cv2.waitKey(0)

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, self.IMG_SIZE)}' % t)

        return predictions


if __name__ == '__main__':

    PATH_model = r"D:\Project\Python_Projects\projects\Yolov5_setup\yolov5\runs\train\exp8\weights\best.pt"
    PATH_img = r"D:\Project\Python_Projects\projects\business_card_automation\Build_1\a_1_0\im.jpg"

    img = cv2.imread(PATH_img)
    OBJ = CardInference(PATH_model)
    OBJ.process(img)
