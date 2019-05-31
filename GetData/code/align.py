import sys
from GetData.code.example import *
import cv2
import time
from PIL import Image
import glob

def createFolder(label):
  import os
  path = "align/"+label
  try:  
    os.makedirs(path)
  except OSError:  
    print ("Creation of the directory %s failed" % path)
  else:  
    print ("Successfully created the directory %s" % path)

def align_Image(label):
  #read path image
  image_list = []
  for filename in glob.glob('image/'+label+"/"+"*.jpg"): #assuming gif
    im=filename
    image_list.append(im)
  createFolder(label)
  x=0
  for path in image_list:
        print(path)
        init(path)
      # image=draw()
        im=getFace()
        for i in range(len(im)):
          x+=1
          try:
            cv2.imwrite("align/" + label + "/" + str(x) + ".jpg", cv2.resize(im[i],(160,160),cv2.INTER_AREA))
          except:
            print("next")
def align_sigleImage(image):
  #read path image
  initImage(image)
  im=getFace()
  return im
def getLocalImage(image):
  begin = time.time()
  initImage(image)
  im=getlocalFace()
  print("align :" + str(time.time()-begin))
  return im
