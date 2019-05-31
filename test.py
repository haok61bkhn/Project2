from GetData.code.webcam import *
from GetData.code.align import *
import glob
from Facenet.test import *
frame=None
def Process():
    one = time.time()
    global frame
    listlocal=getLocalImage(frame)
    
    for local in listlocal:
        x=local[0]
        y=local[2]
        w=local[1]
        h=local[3]
        text=predict(frame[x:w,y:h])
        cv2.rectangle(frame, (y, x), (h, w), (0, 255, 0), 2)
        cv2.putText(frame,text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)


def Process1(x,y,w,h):
     global frame
     text=predict(frame[y:y+h,x:x+w])
     if(text!="0"):
         cv2.imwrite("1.jpg",frame[y:y+h,x:x+w])
         cv2.putText(frame,text, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


   
  
if __name__ == "__main__":
    Init()
  
    cascPath = "GetData/code/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture("video.mp4")
    ftime=time.time()
    begin = time.time()

    while (video_capture.isOpened()):
    # Capture frame-by-frame
    
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,  
         )
    # Draw a rectangle around the faces 
        for (x, y, w, h) in faces:
            Process1(x,y,w,h)
         
      
          
    # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
          break
