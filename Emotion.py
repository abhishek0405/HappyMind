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



def main():

    detection_model_path = 'Haarcascades/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/emotionnetworkweights.hdf5'

    
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","fear", "happy", "neutral", "sad","surprise"]
    limits= [0,0,0,0,0,0,0]
    FINAL_COUNTS= [0,0,0,0,0,0,0]
    
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
       
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

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
                    cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)


        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)
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
print(report)






