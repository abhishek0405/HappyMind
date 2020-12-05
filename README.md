# Healthy Mind

### Inspiration  
With the COVID-19 lockdown, the work culture across the world has changed a lot. People are working from home and attending classes online and this is going to continue for at least a few more months. Moreover, there are no office timings and people are expected to extend their working hours which leads to a lot of mental pressure. It leads to fatigue, irritation, anger and frustration and these are the factors responsible for mental health issues due to workplace stress. 
It therefore becomes really important for people to take proper care of their mental state and take enough breaks to avoid fatigue and stress. 
And that’s why we came up with --- <strong>Healthy Mind </strong>to help people maintain a healthy mental state while working from home. 

### What it does 
It uses a drowsiness detector and an emotion detector to maintain a track of the fatigue and frustration level of the user. In case drowsiness is detected for a significant amount of time, it alerts the user and informs them that it’s time to take a break. At the same time the emotion detector keeps track of the user’s facial expressions and the times when the user looks annoyed, disgusted or anxious. At the end of the session a report is generated for the user which contains the overall mental state and some tips to ease stress and fatigue. In this way it helps the users to introspect and they can attempt to switch to a healthier lifestyle. 

### How We built it 
We trained two models – one for drowsiness detection and the other for emotion detection (built using Tensorflow). We used Convolutional Neural Networks whose architecture can be found in the model architecture folder. The models achieved about 85% accuracy. We then used OpenCV for video streaming and the Haarcascade for detecting faces and eyes after which we had our models predict the emotion of the detected face and the drowsiness in the eyes. Upon detection we maintained a counter for tracking the number of times the user felt drowsy or frustrated by having an appropriate threshold value for determining the prediction with certainty and increasing count. Upon ending the session, a report is generated which contains the statistics of the mental state and suggestions for improvement. We also used the pyttsx3 text to speech library to generate an alarm whenever the user feels drowsy. 


### Challenges we ran into 
The major challenge was to train two models and achieve a decent accuracy. The training process took considerable time since we used a significantly large dataset.  

### Accomplishments that we are proud of 
The fact that we were able to come up with a working model in such a short period of time. The accuracy may not be too high but it is decent enough for the time being.

### Future Scope
We will work on improving the model accuracy by tweaking some parameters in the network layers. It will also be nice to build a better GUI than the current version.

### How to run it
- [x] Clone the repository to your local directory
 - `git clone https://github.com/abhishek0405/NeuralHack.git`
 
- [x] Activate your virtual environment. Follow steps in this link to create your virtual environment : <a href=https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/>Click here</a>

- `pip install virtualenv`
 - `virtualenv env`
 - `env\Scripts\activate`

- [x] Install packages from requirements.txt
- ` pip install -r requirements.txt `

- [x] Run combo.py file 
- ` python combo.py `

- [x] The script starts running and if you want to end the session press ‘q’. It generates a report in the form of a txt file and saves it in your pwd. 

Get the report of your mental state and tips to improve your lifestyle!

###Demo screenshots

Disgust and frustration:
<p  align="center"><img height= "400" width = "800" src = "https://github.com/abhishek0405/NeuralHack/blob/master/images/disgust.jpeg"></p>
<br>
Drowsy
<p  align="center"><img height= "400" width = "800" src = "https://github.com/abhishek0405/NeuralHack/blob/master/images/drowsy.jpeg"></p>
<br>
Happy
<p  align="center"><img height= "400" width = "800" src = "https://github.com/abhishek0405/NeuralHack/blob/master/images/happy.jpeg"></p>
<br>
Mental state chart
<p  align="center"><img height= "400" width = "800" src = "https://github.com/abhishek0405/NeuralHack/blob/master/images/piechart.jpeg"></p>
<br>
Mental state report and suggestions for healthy lifestyle
<p  align="center"><img height= "400" width = "800" src = "https://github.com/abhishek0405/NeuralHack/blob/master/images/report.jpeg"></p>
<br>


### Built With 
-	Python
-	Tensorflow
-	OpenCV
-	pyttsx3

### Team 😊
- [Abhishek Anantharam](https://github.com/abhishek0405) 
- [Vishaka Mohan](https://github.com/vishaka-mohan)

