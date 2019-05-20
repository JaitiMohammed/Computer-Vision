
import cv2
import numpy as np

hand=cv2.CascadeClassifier('data/Hand_haar_cascade.xml')
cap =cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hands=hand.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in hands :
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start ,ycord_end)
        roi_color = img[y:y+h, x:x+w]
        
        img_item="my-image.png"
        cv2.imwrite(img_item ,roi_gray)
        color =(255,0,0)
        stroke =2
        width =x+w
        height =y+h
        cv2.rectangle(img,(x,y),(width,height),color,stroke)
    
    #isplay the resulting frame
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
