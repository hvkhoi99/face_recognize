from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import time
import _datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import dlib
from model import create_model
from align import AlignDlib
import glob
import imutils
import RPi.GPIO as GPIO 
import math
import pyrebase


config = {
  "apiKey": "AIzaSyAKRQLWPnBHvZytF0YiuqXkTbor_knZPsU",
  "authDomain": "fir-657f4.firebaseapp.com",
  "databaseURL": "https://fir-657f4.firebaseio.com",
  "projectId": "fir-657f4",
  "storageBucket": "fir-657f4.appspot.com",
  "messagingSenderId": "855231390742",
  "appId": "1:855231390742:web:159cba5d16121908f14137",
  "measurementId": "G-ZL6DZPEDQX"
};
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()


# INITIALIZE MODELS
nn4_small2 = create_model()

nn4_small2.summary()

nn4_small2.load_weights('weights/nn4.small2.v1.h5')

alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image/*")
print(train_paths)

nb_classes = len(train_paths)
# print(nb_classes)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
print(df_train)

def align_face(face):
    #print(img.shape)
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    #print(bb)
    return alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def align_faces(faces):
    aligned_images = []
    for face in faces:
        #print(face.shape)
        aligned = align_face(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
        
    return aligned_images

def calc_emb_test(faces):
    pd = []
    aligned_faces = align_faces(faces)
    if(len(faces)==1):
        pd.append(nn4_small2.predict_on_batch(aligned_faces))
    elif(len(faces)>1):
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_faces)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
    return np.array(embs)

def calc_distance_euclid(train_embs1, train_embs2):
    distance = 0
    for i in range(len(train_embs1)):
        tmp = pow((train_embs1[i]-train_embs2[i]),2)
        distance = distance + tmp
    distance = math.sqrt(distance)
    return distance


# TRAINING

train_embs = np.load("train_embs.npy",allow_pickle=True)
train_embs = np.concatenate(train_embs)

label2idx = []
for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))



# ANALYSING
# import matplotlib.pyplot as plt
# 
# match_distances = []
# for i in range(nb_classes):
#     ids = label2idx[i]
#     distances = []
#     for j in range(len(ids) - 1):
#         for k in range(j + 1, len(ids)):
#             distances.append(calc_distance_euclid(train_embs[ids[j]], train_embs[ids[k]]))
#     match_distances.extend(distances)
#     
# unmatch_distances = []
# for i in range(nb_classes):
#     ids = label2idx[i]
#     distances = []
#     for j in range(10):
#         idx = np.random.randint(train_embs.shape[0])
#         while idx in label2idx[i]:
#             idx = np.random.randint(train_embs.shape[0])
#         distances.append(calc_distance_euclid(train_embs[ids[np.random.randint(len(ids))]], train_embs[idx]))
#     unmatch_distances.extend(distances)
#     
# _,_,_=plt.hist(match_distances,bins=100)
# _,_,_=plt.hist(unmatch_distances,bins=100,fc=(1, 0, 0, 0.5))

# plt.show()
   
threshold = 0.65

# Setup the GPIO PIN
GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT)
pp= GPIO.PWM(25, 50)



# TEST
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size = (320,240))


countNow = db.child("Users/countNow/count").get().val()
print (countNow)
if(countNow == None):
        countNext = 0
else :
        countNext = int(countNow)
count2 = 0
id = 0
ten = ''
tg = ''

hogFaceDetector = dlib.get_frontal_face_detector()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    test_image = frame.array
    show_image = frame.array


    faceRects = hogFaceDetector(test_image, 0)
#         
    faces = []
#
    
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        face = test_image[y1:y2,x1:x2]
#             
        faces.append(face)
        
        tg = _datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        print("len(faces) = {0}".format(len(faceRects)))
        print(tg)
        
        
#         cv2.rectangle(show_image,(x1,y1),(x2,y2),(0,255,0),2)
#     if(len(faceRects)==0):
#         print("no face detected!")
#     #     continue
    if(len(faces)>0):
        
        test_embs = calc_emb_test(faces)
        test_embs = np.concatenate(test_embs) #mang gom len(faces) phan tu, moi phan tu gom 128 gia tri
    #     print(len(test_embs))
        
        
        match_1 = []  #luu khoang cach euclid cua tung khuon mat trong anh voi cac khuon mat trong dataSet
        for i in range(test_embs.shape[0]): #test_embs.shape[0]: so luong khuon mat
            match = []
            for j in range(len(train_paths)): #len(train_paths); so luong nguoi trong dataSet
                ids = label2idx[j]
                distances = []
                for k in range(len(ids)):
                    distances.append(calc_distance_euclid(test_embs[i], train_embs[ids[k]]))
                match.append(np.asarray(distances))
            match_1.append(np.asarray(match))
        
        count = []   #dem so khuon mat co khoang cach Euclid < threshold cua tung nguoi trong dataSet
        for i in range(len(match_1)):
            tmpp = []
            for j in range(len(match_1[i])):
                dem = 0
                for k in range(len(match_1[i][j])):
                    if(match_1[i][j][k] < threshold):
                        dem = dem + 1
                tmpp.append(dem)
            count.append(np.asarray(tmpp))

        
        
        
        people = []
        phantram = []
        for i in range(test_embs.shape[0]):
            min_distances = []
            for j in range(len(match_1[i])):
                min_distances.append(np.min(match_1[i][j]))
            

            if np.min(min_distances)>threshold:
                people.append("unknown")
                phantram.append(0)
            else:
                for a in range(len(min_distances)):
                    if(min_distances[a] == np.min(min_distances)):
                        res = a
                        print(res)
                        people.append(res)
                        
                        phantram.append(round(np.max(count[i][res])*100/len(match_1[i][res]),1))
                    

        names = os.listdir("image/")
        names_1 = []
        print(phantram)
        
        for p in people:
            if p == "unknown":
                name = "0_unknown"
            else:
                name = names[p]
            names_1.append(name)

                  
        for i,faceRect in enumerate(faceRects):
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            
            info = names_1[i].split('_')
            id = info[0]
            ten = info[1]
            print(id)
            print(ten)
            countNext+= 1
            camera.start_preview()
            sleep(1)
            camera.capture('/home/pi/Desktop/keras-and-dlib-master/img/'+str(countNext)+'.jpg')
            camera.stop_preview()
            
            data = {"Users/user_0" + str(countNext) + "/Id" : str(id)}
            db.child("").child().update(data)
            data = {"Users/user_0" + str(countNext) + "/Name" : ten}
            db.child("").child().update(data)
            data = {"Users/user_0" + str(countNext) + "/Datetime" : tg}
            db.child("").child().update(data)
            
            data = {"Users/countNow/count" : str(countNext)}
            db.child("").child().update(data)
            
            path_on_cloud = str(countNext)+'.jpg'
            path_local = 'img/'+str(countNext)+'.jpg'

            storage.child(path_on_cloud).put(path_local)
            
            
            cv2.rectangle(show_image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(show_image,ten,(x1,y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
            if(phantram[i]>0):
                cv2.putText(show_image,str(phantram[i]),(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),1)
                #den
                channel = 17
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(channel, GPIO.OUT)
                GPIO.output(channel, GPIO.HIGH)
                time.sleep(4)
                GPIO.output(channel, GPIO.LOW)
                time.sleep(1)

                #cua
                pp.start(13)
                pp.ChangeDutyCycle(12)
                time.sleep(5)
                pp.ChangeDutyCycle(2)
                time.sleep(1)
                pp.ChangeDutyCycle(0)
          

    cv2.imshow("image", show_image)
    
            
    if cv2.waitKey(1) & 0xff == ord("q"):
        exit()
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
            


