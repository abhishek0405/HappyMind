from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
import pyttsx3
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def speak(audio):   
    engine.say(audio)  
    engine.runAndWait()




def main():
    

    detection_model_path = 'Haarcascades/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/emotionnetworkweights.hdf5'
    face = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('Haarcascades/haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('Haarcascades/haarcascade_righteye_2splits.xml')
    model = load_model('models/drowsy1.h5')
    face_detection = cv2.CascadeClassifier(detection_model_path)

    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","fear", "happy", "neutral", "sad",
     "surprise"]
    sleep_count=0
    sleep_lim = 0

    right_pred=[100]
    left_pred=[100]
    limits= [0,0,0,0,0,0,0]
    FINAL_COUNTS= [0,0,0,0,0,0,0]

    
    #cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        height,width = frame.shape[:2]
       
        #frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        #faces1 = face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        left_eyes = leye.detectMultiScale(gray)
        right_eyes =  reye.detectMultiScale(gray)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                      
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)


            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            index = EMOTIONS.index(label)
            print("label is",label)
            print("index is",index)
            limits[index]+=1
            if(limits[index]>=15):
                limits[index]=0
                FINAL_COUNTS[index]+=1
        else: continue


        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                  


                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frame, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)

        #for (x,y,w,h) in faces1:
            #cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 1 )

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

        
        
        
        

        #cv2.imshow('your_face', frameClone)
        cv2.imshow('pro',frame)
        #cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

         
        
        
            

            
    
    camera.release()
    cv2.destroyAllWindows()
    return FINAL_COUNTS


final_result=main()
EMOTIONS = ["angry" ,"disgust","fear", "happy", "neutral", "sad","surprise"]
report={}
for i in range(len(final_result)):
    if(EMOTIONS[i]=="surprise"):
        pass
    else:
        report[EMOTIONS[i]] = final_result[i]


image = np.zeros((512,512,3), np.uint8)
cv2.putText(image,'Report:',(30,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
cv2.imshow('report', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(report)









