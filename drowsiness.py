import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import time
import pyttsx3


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def speak(audio):   
    engine.say(audio)  
    engine.runAndWait()



face = cv2.CascadeClassifier('Haarcascades\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('Haarcascades\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('Haarcascades\haarcascade_righteye_2splits.xml')



model = load_model('models/drowsy1.h5')

cap = cv2.VideoCapture(0)

sleep_count=0
sleep_lim = 0

right_pred=[100]
left_pred=[100]

while True:
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eyes = leye.detectMultiScale(gray)
    right_eyes =  reye.detectMultiScale(gray)

    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 1 )

    for (x,y,w,h) in right_eyes:
        right_eye=frame[y:y+h,x:x+w]
        
        right_eye = cv2.cvtColor(right_eye,cv2.COLOR_BGR2GRAY)
        right_eye = cv2.resize(right_eye,(24,24))
        right_eye= right_eye/255
        right_eye=  right_eye.reshape(24,24,-1)
        right_eye = np.expand_dims(right_eye,axis=0)
        right_pred = model.predict_classes(right_eye)
        
        break

    for (x,y,w,h) in left_eyes:
        left_eye=frame[y:y+h,x:x+w]
        
        left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2GRAY)  
        left_eye = cv2.resize(left_eye,(24,24))
        left_eye= left_eye/255
        left_eye=left_eye.reshape(24,24,-1)
        left_eye = np.expand_dims(left_eye,axis=0)
        left_pred = model.predict_classes(left_eye)
        
        break

    if right_pred[0]==0 and left_pred[0]==0:
        sleep_count += 1
        
    else:
        sleep_count -= 1
        
    
        
    if sleep_count < 0:
        sleep_count=0
        
    cv2.putText(frame,'Sleep count:'+str(sleep_count),(100,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)

    
    if sleep_count >= 30:
        sleep_count = 0

        sleep_lim += 1
        print(sleep_lim)
    
        try:
            
            speak('Wake up')
            
            
        except:  
            pass

    if sleep_lim >=10:

        sleep_lim = 0

        try:
            
            speak('It\'s time to take a break!')
            
            
        except:  
            pass

        
        
    cv2.imshow('pro',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
