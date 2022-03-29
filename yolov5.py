import os
import sys
from pathlib import Path
from utils.torch_utils import select_device, time_sync
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,xywhn2xyxy)

from models.common import DetectMultiBackend
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib
from classify import Classify

class YOLOv5(object):
    _defaults={
        "weights": ROOT /"weight/best39.pt",
        "imgsz": 640,
        "iou_thres":0.45,
        "conf_thres":0.25,
        "data":ROOT /"data/TACO.yaml"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    # 初始化操作，加载模型
    def __init__(self, device='cpu', **kwargs):
        self.__dict__.update(self._defaults)
        self.device = select_device(device)
        print(self.device)

        # self.half = self.device != "cpu"

        self.model= DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)  # check img_size
        # if self.half:
        #     self.model.half()  # to FP16

    def get_featuremap(self,inImg):
        img = letterbox(inImg, new_shape=self.imgsz)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred,features = self.model(img, augment=False) #17/20/23 =>128/256/512
        return features
    def process_features(self,pre_features,post_features,bbox_xywh):

        print(bbox_xywh)
        bbox_xywh=torch.tensor(bbox_xywh)
        for index in (4,6,10,14):
            pre_feature=torch.squeeze(pre_features[index],dim=0)
            post_feature=torch.squeeze(post_features[index],dim=0)

            shape=pre_feature.shape[1:]
            pre_feature=torch.permute(pre_feature,dims=[1,2,0])
            post_feature=torch.permute(post_feature,dims=[1,2,0])
            reflect=xywhn2xyxy(bbox_xywh,w=shape[1],h=shape[0])
            dislist=[]
            for cor in reflect:
                x,y=int(cor[0]),int(cor[1])
                result_pre=pre_feature[x][y]
                result_post=post_feature[x][y]
                dis=torch.sum((result_post-result_pre)**2).item()
                dislist.append(dis)
                print(dis)
            plt.plot(range(len(dislist)),dislist)
            plt.title("feature distance")
            plt.show()
            print(reflect)


    def infer_single(self,inImg):
        img = letterbox(inImg, new_shape=self.imgsz)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred, _ = self.model(img, augment=False)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=True)

        bbox_xyxy = []
        bbox_xywh = []
        confs = []
        cls_ids = []
        xyxys = torch.Tensor([])
        confss = torch.Tensor([])
        gn = torch.tensor(inImg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # 解析检测结果
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # 将检测框映射到原始图像大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], inImg.shape).round()
                # 保存结果
                for *xyxy, conf, cls in reversed(det.cpu()):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bbox_xywh.append(xywh)
                    cordinat = []
                    for item in xyxy:
                        cordinat.append(item.item())
                    bbox_xyxy.append(cordinat)
                    confs.append(conf.item())
                    cls_ids.append(int(cls.item()))
                bbox_xywh = np.asarray(bbox_xywh)
                xyxys = np.asarray(bbox_xyxy, np.int32)
                confss = np.asarray(confs)
                cls_ids = np.asarray(cls_ids)
        detail=[]
        if(len(xyxys)>0):
            #补充fenleit
            patch=self.get_patch(img=inImg,anchors=xyxys)

            detail=self.classifier(patch)


        return (xyxys, confss, cls_ids,detail)
    def infer(self,inImg,preImg):
        print(self.weights)
        img = letterbox(inImg, new_shape=self.imgsz)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img =  img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred,_= self.model(img, augment=False)
        # pre_features=self.get_featuremap(preImg)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=True)

        bbox_xyxy = []
        bbox_xywh = []
        confs = []
        cls_ids = []
        xyxys=torch.Tensor([])
        confss=torch.Tensor([])
        gn = torch.tensor(inImg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # 解析检测结果
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # 将检测框映射到原始图像大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], inImg.shape).round()
                # 保存结果
                for *xyxy, conf, cls in reversed(det.cpu()):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    bbox_xywh.append(xywh)
                    cordinat=[]
                    for item in xyxy:
                        cordinat.append(item.item())
                    bbox_xyxy.append(cordinat)
                    confs.append(conf.item())
                    cls_ids.append(int(cls.item()))
                bbox_xywh=np.asarray(bbox_xywh)
                xyxys = np.asarray(bbox_xyxy,np.int32)
                confss =np.asarray(confs)
                cls_ids = np.asarray(cls_ids)
        # self.process_box(xyxys)
        # self.process_features(pre_features, post_features,bbox_xywh)
        picks=self.detect_newadd(preImg=preImg,postImg=inImg,bboxes=xyxys)

        # if(len(xyxys[picks])>0):
        #     #补充fenleit
        #     patch=self.get_patch(img=inImg,anchors=xyxys[picks])
        #
        #     detail=self.classifier(patch)
        #     print(detail)



        return (xyxys[picks],confss[picks],cls_ids[picks])
        # return xyxys,confss,cls_ids
    def process_box(self,bboxs_xyxy):
        from torchvision.ops import box_iou
        bboxs_xyxy=torch.tensor(bboxs_xyxy)
        resultiou = box_iou(bboxs_xyxy, bboxs_xyxy)  # nXn
        print(resultiou)
        inset= {}
        for i in range(len(bboxs_xyxy)):
            inset[i]=[]
            for j in range(i,len(bboxs_xyxy)):
                if(i==j):
                    continue
                if(resultiou[i][j]!=0.0):
                    inset[i].append(j)
        newbboxs=[]
        for i in range(len(bboxs_xyxy)):
            newbox=bboxs_xyxy[i].clone().detach()
            for boxindex in inset[i]:
                if(box_iou(torch.unsqueeze(bboxs_xyxy[boxindex],dim=0),torch.unsqueeze(newbox,dim=0))[0][0]!=0.0):
                    newbox=self.adjust_two_box(bboxs_xyxy[boxindex],newbox)
            newbboxs.append(newbox)
        return torch.stack(newbboxs)

    def isinside(self,ret,point):
        if(point[0]>ret[0] and point[0]<ret[2] and point[1]>ret[1] and point[1]<ret[3]):
            return True
        else:
            return False
    def adjust_two_box(self,box1_,box2_):
        print(box1_)
        print(box2_)
        #fix box1
        box1= box1_.clone().detach()
        box2 = box2_.clone().detach()
        if(self.isinside(box1_,box2_[2:])):
            box1[1]=box2_[3]
            box2[3]=box1_[1]
            return box2
            pass
        elif (self.isinside(box1_,box2_[[2,1]])):
            box1[3] = box2_[1]
            box2[1] = box1_[3]
            return box2
            pass
        elif(self.isinside(box1_,box2_[[0,3]])):
            box1[1] = box2_[3]
            box2[3] = box1_[1]
            return box2
            pass
        elif(self.isinside(box1_,box2_[:2])):
            box1[3] = box2_[1]
            box2[1] = box1_[3]
            return  box2
            pass
        else:
            self.adjust_two_box(self, box2_, box1_)

        pass
    def get_patch(self,img,anchors):
        hight = img.shape[0]
        weight = img.shape[1]
        patch=[]
        for cropitem in anchors:
            crop = img[cropitem[1]:cropitem[3], cropitem[0]:cropitem[2]].copy()
            # print(crop)
            # cv2.imshow("cropped", crop)
            # cv2.waitKey(0)
            patch.append(crop)

        return patch
    def get_ccv(self,S:np.ndarray,T:np.ndarray):
        w=T.shape[1]
        h = T.shape[0]
        res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
        print(res[0][0])


        return res[0][0]

    def count_boxsize(self,bboxes):
        boxsize=[]
        for box in bboxes:
            boxsize.append((box[2]-box[0])*(box[3]-box[1]))
        minindex=np.argmin(boxsize)
        return minindex
        pass
    def detect_newadd(self, preImg,postImg,bboxes):

        if(bboxes is None):
            return None
        boxsizes=self.count_boxsize(bboxes)
        patches_post = self.get_patch(img=postImg, anchors=bboxes)
        patches_mappre = self.get_patch(img=preImg, anchors=bboxes)

        ccvs=[]
        for post, pre in zip(patches_post, patches_mappre):
            # cv2.imshow("cropped", post)
            # cv2.waitKey(0)
            # cv2.imshow("cropped", pre)
            # cv2.waitKey(0)
            ccv=self.get_ccv(post,pre)

            ccvs.append(ccv)
        ccvs = np.array(ccvs)
        # plt.plot(range(len(ccvs)),-1*ccvs)
        # plt.title("RGB similarity")
        # plt.show()
        ccvs=np.array(ccvs)
        picks=[]
        for index,ccv in enumerate(ccvs) :
            if(ccv<np.mean(ccvs)):
                picks.append(index)
        newbboxs=bboxes[picks]



        return picks
    def classifier(self,imglist):
        classifyhead=Classify()
        results=classifyhead.infer_batch(imglist)
        return results



def plot_one_box(x, img, color=None, label="person", line_thickness=None):
    import  random
    """ 画框,引自 YoLov5 工程.
    参数:
        x:      框， [x1,y1,x2,y2]
        img:    opencv图像
        color:  设置矩形框的颜色, 比如 (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
if __name__ == '__main__':

    det = YOLOv5()
    base="data/testimg/clas/"

    # iImage = cv2.cvtColor(cv2.imread(base + "img_18121.jpg"),code=cv2.COLOR_BGR2RGB)
    # boxs, confss, cls_ids = det.infer_single(inImg=iImage)
    # if boxs is not None:
    #     for i, box in enumerate(boxs):
    #         plot_one_box((box), iImage, label=str(cls_ids[i]))
    #         plt.imshow(iImage)
    #         plt.show()
    # pre_iImage=cv2.imread(base+"8804-pre.jpg")
    # post_iImage = cv2.imread(base + "8804-post.jpg")
    # boxs,confss,cls_ids = det.infer_single(preImg=pre_iImage, inImg=post_iImage)
    # if boxs is not None:
    #     for i, box in enumerate(boxs):
    #         plot_one_box((box), post_iImage, label=str(cls_ids[i]))
    #         plt.imshow(post_iImage)
    #         plt.show()