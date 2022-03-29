import requests
import json
import random
import cv2

if __name__ == '__main__':

    path="data/testimg/clas/fimg_2798.jpg"

    Frame = open(path, "rb").read()
    request_input = {'img': Frame}
    result = requests.post('http://127.0.0.1:7000/inferclassify', files=request_input).json()
    print(result)
    if result['success']:
        print(result['big_category'])
    else:
        print("error")