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

global flag
flag=0


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
        #height,width = frame.shape[:2]
       
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
            #print("label is",label)
            #print("index is",index)
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
        
        cv2.putText(frame,'Sleep count:'+str(sleep_count),(100,400-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1,cv2.LINE_AA)

    
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
            flag=1
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

s=''
win_text=''
for key,val in report.items():
    if(key=='angry' and val>=5):
        s1='''
        You have been angry for a significant time during this work session.Think back what it was that made you lose your temper.
        Some tips to help you control your anger:
        1. BREATHE DEEPLY AND COUNT TO 10.
        
        2. TALK TO SOMEONE YOU CAN TRUST.
        
        3. RECOGNISE YOUR PERSONAL TRIGGERS AND TRY TO WORK ON THEM.
        
        Stay Calm Stay happy :)
        --------------------------------------------------------------------------------------------------------------------
        
        '''
        s1=s1.strip()
        s+=s1+ '\n'
        if win_text=='':
            win_text = 'You have been angry for a significant time during this work session.Think back what it was that made you lose your temper.'
        
    elif(key=='disgust' and val>=5):
        s1='''
            Feeling disgust for something, or worse, someone, is one of the most difficult emotional states for anyone to control.
            But this is something you must work on based on this session.
            Here are a few tips:
            
            1. Notice when judgmental thoughts pop into your head and try to control yourself.
            
            2. Remember to breathe as calming breaths also engage your frontal lobe and will make you feel better.
            
            3.  Talk to someone you trust about your feelings. 
           ----------------------------------------------------------------------------------------------------------------------
        '''
        s1=s1.strip()
        s+=s1 + '\n'

        if win_text=='':
            win_text = 'Feeling disgust for something, or worse, someone, is one of the most difficult emotional states for anyone to control.'
        
    elif(key=='fear' and val>=5):
        s1='''
            Based on this session, a feeling of anxiety and fear was captured for a significant amount of time.Think carefully what could the reason be.
            Was it becuase you were running late on deadlines,or perhaps took too much of work for yourself?
            Try to answer these questions and figure out a way to control your anxiety.
            Here are a few tips:
            
            1. Stick to a routine
                Having a routine — and sticking to it — can help manage the symptoms.
                
            2. Exercise and listen to your body
                Exercise is excellent for clearing your head.
                
            3. Make time for yourself
                Make time to unwind. Do things you enjoy: Baking, meditating, reading, journaling, or listening to music.
            --------------------------------------------------------------------------------------------------------------------
        '''
        s1=s1.strip()
        s+=s1 + '\n'

        if win_text=='':

            win_text = 'Based on this session, a feeling of anxiety and fear was captured for a significant amount of time.Think carefully what could the reason be.'
    elif(key=='happy' and val<5):
        s1='''
            Statistics prove that being happy while working is something most succesful people have in common.The signs of happiness detected in this session were less than the average value.
            
            Here are few stips to stay happy while waorking:
            
            1 Set a schedule.
                When you work from home, it is tempting to sleep late and then work until whenever, but this is not the path to productivity. Our brains like regularity, so set your alarm clock to get up at the same time every day (preferably early).
            
            2 Make social plans for after work.
                Working from home has huge benefits, but let’s face it—you get a little lonely. If you are going to go on social media, schedule it into your day, such as “10 am: 5 min. Facebook break.” This will help you stay balanced.
                
            3.Take advantage of not being in office:
               You get to be in your happy place all day, so make the most of it. With no coworkers to quibble over your musical taste, you can play tunes in the background while you work.
               --------------------------------------------------------------------------------------------------------------------
        '''
        s1=s1.strip()
        s+=s1 + '\n'

        if win_text=='':
            win_text = 'Statistics prove that being happy while working is something most succesful people have in common.The signs of happiness detected in this session were less than the average value.'
    elif(key=='sad' and val>=5):
        s1='''
            Being sad is a normal part of life. It may come as a result of environmental factors. You facial expressions indicated sadness during this session.
            Here are a few tips to help you:
            
            1. Call a friend
                You can even have a friend record a message about their day and send it your way. And you can do the same.
                
            2. Create a daily schedule
                It can be easy to lose track of time when you aren’t in an office. Creating a schedule for the day not only helps you get your tasks done, but it also pencils in opportunities to take breaks to maintain your mental health.
                
            3. Go for a walk
                Going for a walk benefits your mental health as well as your physical health.
                --------------------------------------------------------------------------------------------------------------------
        '''
        s1=s1.strip()
        s+=s1 + '\n'
        
        win_text='Being sad is a normal part of life. It may come as a result of environmental factors. You facial expressions indicated sadness during this session.'

    elif(flag==1):
        s1='''
                You have shown signs of sleepiness in this work session.This will have a negative impact in your overall work performance.
                Here are some tips to stay fresh during work:

                1.Stick to a set sleep schedule.Try to have atleast 7-8 hours of sleep daily

                2.Eat for energy. Having your tummy full just before work sure would make you feel sleepy. Try to eat till only you are 80% full.

                3.Drink more water: It is proven that drinking water frequently is extremely good for your body and will help you stay fresh.

                
            '''
        s1=s1.strip()
        s+=s1+'\n'

        win_text='You have shown signs of sleepiness in this work session.This will have a negative impact in your overall work performance.'

        
      

newresult=[]
for item in final_result:
    newresult.append(item+1)

print(newresult[:-1])
print(EMOTIONS[:-1])


fig1, ax1 = plt.subplots()
ax1.pie(newresult[:-1], labels=EMOTIONS[:-1], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('report.pdf')
     
s2= s.strip()        

file = open('report.txt','w') 
file.write(s2)
file.close() 
      
image = np.zeros((1024,2048,3), np.uint8)
cv2.putText(image,'Report:',(30,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
cv2.putText(image,win_text,(30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
cv2.putText(image,'Hope you had a productive work session.Refer the report.pdf and report.txt files to get a detailed insight',(30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

cv2.imshow('report', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(report)









