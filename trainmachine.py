import os
import cv2
import numpy as np


cascade_path = r'C:/Users/Pc/Documents/hass.xml'
haar_cas = cv2.CascadeClassifier(cascade_path)
p=[]
for i in os.listdir(r'C:/Users/Pc/Desktop/padugeevitham'):
    p.append(i)

Dir = r'C:/Users/Pc/Desktop/padugeevitham'
features =[]
lables = []

def create_train():
    for Person in p:
        path = os.path.join(Dir,Person)
        lable = p.index(Person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv2.imread(img_path)
            gray= cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
               faces_roi =gray[y:y+h, x:x+w]
               features.append(faces_roi)
               lables.append(lable)

create_train()

features = np.array(features , dtype ='object')
lables = np.array(lables)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


face_recognizer.train(features,lables)
np.save('features.npy', features)
np.save('lables.npy' , lables)

face_recognizer.save('face_trained5_0.yml')





