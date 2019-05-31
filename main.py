# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
from GetData.code.webcam import *
from GetData.code.align import *
import glob
from Facenet.test import *
from Facenet.train_main import *
draw=None
def Process1(x,y,x1,y1):
     global draw
     text=predict(draw[y:y1,x:x1])
    
     if(True):
         cv2.putText(draw,text, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
         cv2.rectangle(draw, (x, y), (x1,y1), (0, 255, 0), 2)

def run():
    global draw
    Init()
    detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(), num_worker = 8 , accurate_landmark = True)
    #camera = cv2.VideoCapture("video.mp4")
    camera = cv2.VideoCapture(0)
    while True:
        grab, frame = camera.read()
        img = cv2.resize(frame, (500,500))

        t1 = time.time()
        results = detector.detect_face(img)
        t2 = time.time()
        print(time.time()-t2)
        draw = img.copy()
        if results != None:
            total_boxes = results[0]
            points = results[1]
            for b in total_boxes:
                Process1(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        cv2.imshow("detection result", draw)
        cv2.waitKey(20)
if __name__ == "__main__":
    getData("Van")
    train()
    run()
