import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
#hand=cv2.CascadeClassifier('data/hand.xml')
cap =cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start ,ycord_end)
        roi_color = img[y:y+h, x:x+w]
        
        img_item="my-image.png"
        cv2.imwrite(img_item ,roi_gray)
        color =(255,0,0)
        stroke =2
        width =x+w
        height =y+h
        cv2.rectangle(img,(x,y),(width,height),color,stroke)
        
        #eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    #isplay the resulting frame
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

cap.release()
cv2.destroyAllWindows()
