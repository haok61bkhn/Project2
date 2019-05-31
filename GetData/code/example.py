#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from GetData.code.mtcnn import MTCNN
result=None
image=None
image1=None
def getFace():
    global result
    imag=[]
    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints'] 
        ima=image1[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
        imag.append(ima)
    return imag
def getlocalFace():
    global result
    imag=[]
    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']
        imag.append([bounding_box[1],bounding_box[1] + bounding_box[3],bounding_box[0],bounding_box[0]+bounding_box[2]]) 
    return imag

def draw():
    global result
    for i in range(len(result)):
     bounding_box = result[i]['box']
     keypoints = result[i]['keypoints'] 
     cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                   (0,155,255),
                2)
     cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
     cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
     cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
     cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
     cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)    
    return image
 
def initImage(image):
    global result,image1
    detector = MTCNN()
    image1 = image.copy()
    result = detector.detect_faces(image)
   

def init(url):

    global result,image,image1
    detector = MTCNN()
    image = cv2.imread(url)
    image1 = image.copy()
    result = detector.detect_faces(image)
   

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.

    # cv2.imwrite("ivan_drawn.jpg", image)
    # cv2.imshow("ivan_drawn.jpg", image)
    # cv2.waitKey(0)
if __name__ == "__main__":
    init("family.jpg")
    image=draw()
    print(image)
    cv2.imshow("image",image)
    cv2.waitKey(0)
