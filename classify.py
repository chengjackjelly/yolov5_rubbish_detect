import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.torch_utils import select_device
from torchvision import transforms
import matplotlib
from models.main import get_res_model_my
import json
import torch
from PIL import Image
import numpy as np
import cv2
matplotlib.use('Qt5Agg')
with open("detail/id_to_name48.json",'r') as load_f:
    id_to_name = json.load(load_f)
with open("detail/detail_to_big48.json", 'r') as load_f:
    detail_to_big = json.load(load_f)
class Classify(object):
    _defaults={
        "weights": ROOT /"weight/gabtest3_50_withval_cbamAA_au2.pth",
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    # 初始化操作，加载模型
    def __init__(self, device='0', **kwargs):
        self.__dict__.update(self._defaults)
        self.device = select_device(device)
        self.model=get_res_model_my(num_classes=48,res_layer=50,model_path=self.weights)
    def infer_single(self,inImg):
        #test
        self.device='cpu'
        #图像数据转为tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        inImg = Image.fromarray(inImg).convert('RGB')
        img_as_input=transform(inImg)
        img_as_input=torch.unsqueeze(img_as_input,dim=0)
        model=self.model
        model.eval()
        with torch.no_grad():
            _, _, _, _, logits,_ = model(img_as_input.to(self.device))
        logits=torch.squeeze(logits, dim=0)
        logit=logits.argmax(dim=-1)
        print(logit)
        result=id_to_name[str(logit.cpu().item())]
        return result
    def infer_batch(self,inImgs):
        #test
        self.device = 'cpu'
        imglist=[]
        detail_labels=[]
        # 图像数据转为tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        for im in inImgs:
            imglist.append(transform(Image.fromarray(im).convert('RGB')))
        imglist=torch.stack(imglist)
        model = self.model
        model.eval()
        with torch.no_grad():
            _, _, _, _, logits, _ = model(imglist.to(self.device))
        for logit in logits:
            logit = logit.argmax(dim=-1)
            result = id_to_name[str(logit.cpu().item())].split('/')[0]
            detail_labels.append(result)
        return detail_labels

if __name__ == '__main__':
    path = "D:/dataset_garb/garbagedir/dir/可回收垃圾/饮料瓶/img_18015.jpg"
    Frame = open(path, "rb").read()
    image = Frame
    imBytes = np.frombuffer(image, np.uint8)

    iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
    classifyhead=Classify()
    print(classifyhead.infer_single(inImg=iImage))