import requests
import json
import random
import cv2
import argparse
# 定义工具方法，在原始图像上画框
map_name={'0':"dry",'1':"wet",'2':"bag"}
def plot_one_box(x, img, color=None, label="person", line_thickness=None):
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
        t_size = cv2.getTextSize(map_name[label], 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            map_name[label],
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post', nargs='+', type=str, default='data/testimg/prepost/27427-post.jpg', help='pre img path(s)')
    parser.add_argument('--pre', nargs='+', type=str, default='data/testimg/prepost/27427-pre.jpg', help='post img path(s)')
    parser.add_argument('--plot', nargs='+', type=bool, default=True,
                        help='pre img path(s)')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    import os
    # imgs=os.listdir(base+"prepost/")
    opt = parse_opt()
    prepath=opt.pre
    postpath=opt.post
    preFrame = open(prepath, "rb").read() #以二进制格式读入图像数据。
    postFrame = open(postpath, "rb").read()
    request_input = {'preimg': preFrame, 'postimg': postFrame}

    result = requests.post('http://127.0.0.1:7000/infer', files=request_input).json()
    print(result)
    if result['success']:
        if (result['zero_detect'] != True):
            if(opt.plot):
                boxs = result["box"]
                img = cv2.imread(postpath)
                ids = result["classid"]
                if boxs is not None:
                    for i, box in enumerate(boxs):
                        plot_one_box((box), img, label=str(ids[i]))
                cv2.imwrite("result/result.jpg",img)
        else:
            print("no new object detected")

    else:
        print("error")
    # for im in imgs:
    #     if(im.split('.')[0].split('-')[-1]=='pre'):
    #         post=im.split('.')[0].split('-')[0]+"-post.jpg"
    #         pre=im
    #         prepath=base+"prepost/"+pre
    #         postpath=base+"prepost/"+post
    #
    #         preFrame = open(prepath, "rb").read()
    #         postFrame = open(postpath, "rb").read()
    #         savepath=base+"prepostresult2/"
    #
    #
    #
    #         request_input = {'preimg': preFrame,'postimg':postFrame}
    #
    #         result = requests.post('http://127.0.0.1:7000/infer',files=request_input).json()
    #         if result['success']:
    #             if(result['zero_detect']!=True):
    #                 print(result["classid"])
    #                 print(result["box"])
    #                 boxs=result["box"]
    #                 img=cv2.imread(postpath)
    #                 ids=result["classid"]
    #                 if boxs is not None:
    #                     for i, box in enumerate(boxs):
    #                         plot_one_box((box), img, label=str(ids[i]))
    #                     cv2.imwrite(savepath+post, img)
    #                     cv2.waitKey(0)
    #             else:
    #                 img = cv2.imread(postpath)
    #                 cv2.imwrite(savepath + post, img)
    #                 cv2.waitKey(0)
    #
    #         else:
    #             print("error")