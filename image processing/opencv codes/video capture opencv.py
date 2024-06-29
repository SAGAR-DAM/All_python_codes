# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:51:25 2023

@author: sagar
"""

import cv2
cap=cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3,600)
cap.set(4,400)

print(cap.get(3))
print(cap.get(4))

while(cap.isOpened()):
    ret,frame=cap.read()
    if ret==True:
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        text='Width: '+str(cap.get(3))+"  "+" Height: "+str(cap.get(4))
        frame=cv2.putText(frame,text,(10,50),font,1,(0,0,255),1,cv2.LINE_AA)
        #frame=cv2.line(frame,(0,0),(200,200),(0,0,255),5)
        #grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

for v in dir():
    exec('del '+ v)
    del v