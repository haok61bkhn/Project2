import cv2
import sys
import time;
from GetData.code.align import *
def createFolder(label):
  import os
  path = "image/"+label
  try:  
    os.makedirs(path)
  except OSError:  
    print ("Creation of the directory %s failed" % path)
  else:  
    print ("Successfully created the directory %s" % path)
def getData(label):
    createFolder(label)
    cascPath = "GetData/code/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    ftime=time.time()
    begin = time.time()

    while (time.time()-begin<=30):
    # Capture frame-by-frame
    
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,  
         )

    # Draw a rectangle around the faces 
        for (x, y, w, h) in faces:
          if (time.time() - ftime >= 0.5):
             ftime=time.time()
             imageName = str("image/"+label+"/"+time.strftime("%Y_%m_%d_%H_%M_%S") + '.jpg')
             cv2.imwrite(imageName, frame)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
          
    # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    video_capture.release()
    cv2.destroyAllWindows()
    align_Image(label)
# When everything is done, release the capture
