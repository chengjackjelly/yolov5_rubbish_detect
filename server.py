import time

from flask import Flask, request, jsonify

import numpy as np
import cv2

from yolov5 import YOLOv5
from classify import Classify
app = Flask(__name__)
det=YOLOv5()
map_name={'0':"dry",'1':"wet",'2':"bag"}

#该接口功能：传入投放前preimg 投放后postimg 返回增量检测结果
@app.route("/infer", methods=["POST"])
def predict():
    result={"success":False}
    if request.method == "POST":

        if request.files.get("preimg") is not None and request.files.get("postimg") is not None:
            print("here")

            start = time.time()
            post_image = request.files["postimg"].read()
            pre_image=request.files["preimg"].read()
            post_imBytes = np.frombuffer(post_image, np.uint8)
            pre_imBytes = np.frombuffer(pre_image, np.uint8)
            post_iImage = cv2.imdecode(post_imBytes, cv2.IMREAD_COLOR)
            pre_iImage=cv2.imdecode(pre_imBytes, cv2.IMREAD_COLOR)

            outs = det.infer(preImg=pre_iImage,inImg=post_iImage)
            print("duration: ", time.time() - start)
            if(len(outs[2])!=0):
                result["box"] = outs[0].tolist()

                result["conf"] = outs[1].tolist()
                result["classid"] =outs[2].tolist()
                result["classname"]=[]
                for item in result["classid"]:
                    result["classname"].append(map_name[str(item)])
                result["zero_detect"] = False
                result["success"]=True
            else:

                result["success"] = True
                result["zero_detect"]=True
            # result["detect"]=ans
            # print(result)

        else:
            print("none")
    return jsonify(result)

#该接口功能：对单张图片进行垃圾的检测、识别
@app.route("/infersingle", methods=["POST"])
def predict_single():
    result={"success":False}
    if request.method == "POST":

        if request.files.get("img") is not None :
            print("here")
            start = time.time()
            image = request.files["img"].read()

            imBytes = np.frombuffer(image, np.uint8)
            iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
            outs = det.infer_single(inImg=iImage)
            print("duration: ", time.time() - start)
            if(len(outs[2])!=0):
                result["box"] = outs[0].tolist()

                result["conf"] = outs[1].tolist()
                result["classid"] =outs[2].tolist()
                result["detail"]=outs[3]
                result["classname"]=[]
                for item in result["classid"]:
                    result["classname"].append(map_name[str(item)])
                result["zero_detect"] = False
                result["success"]=True
            else:

                result["success"] = True
                result["zero_detect"]=True
            # result["detect"]=ans
            # print(result)

        else:
            print("none")
    return jsonify(result)


#该接口功能：单张图片进行分类，供企业测试模型精度
@app.route("/inferclassify", methods=["POST"])
def classify():
    result={"success":False}
    if request.method == "POST":

        if request.files.get("img") is not None :
            print("here")

            image = request.files["img"].read()

            imBytes = np.frombuffer(image, np.uint8)

            iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)

            classifyhead=Classify()
            try:
                category=classifyhead.infer_single(iImage)
                result['success'] = True
                result['big_category'] = category.split('/')[0]
                result['detail_category'] = category.split('/')[1]

            except:
                result['success']=False


    return jsonify(result)
if __name__ == "__main__":
    print(("* Loading yolov5 model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='127.0.0.1', port=7000)