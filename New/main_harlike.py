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
import math




# INITIALIZE MODELS
nn4_small2 = create_model()

nn4_small2.summary()

nn4_small2.load_weights('weights/nn4.small2.v1.h5')

alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image/*")
print(train_paths)

nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
# print(df_train.head())

# PRE-PROCESSING
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_face(face):
    #print(img.shape)
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    #print(bb)
    return alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
  
def load_and_align_images(filepaths):
    aligned_images = []
    for filepath in filepaths:
        #print(filepath)
        img = cv2.imread(filepath)
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
            
    return np.array(aligned_images)
    
def calc_embs(filepaths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(filepaths[start:start+batch_size])
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)

    return np.array(embs)
    
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
label2idx = []

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))

train_embs = calc_embs(df_train.image)
np.save("train_embs.npy", train_embs)

train_embs = np.concatenate(train_embs)

# ANALYSING
import matplotlib.pyplot as plt

match_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
    distances = []
    for j in range(len(ids) - 1):
        for k in range(j + 1, len(ids)):
            distances.append(calc_distance_euclid(train_embs[ids[j]], train_embs[ids[k]]))
    match_distances.extend(distances)
    
unmatch_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
    distances = []
    for j in range(30):
        idx = np.random.randint(train_embs.shape[0])
        while idx in label2idx[i]:
            idx = np.random.randint(train_embs.shape[0])
        distances.append(calc_distance_euclid(train_embs[ids[np.random.randint(len(ids))]], train_embs[idx]))
    unmatch_distances.extend(distances)
    
_,_,_=plt.hist(match_distances,bins=100)
_,_,_=plt.hist(unmatch_distances,bins=100,fc=(1, 0, 0, 0.5))

plt.show()
   
threshold = 0.65

# TEST
camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size = (320,240))

# test_image = cv2.imread("test_image/00000.png")
# show_image = test_image.copy()

tg = ''
ten = ''

#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    test_image = frame.array
    show_image = frame.array


    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    faceRects = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (100, 100), flags = cv2.CASCADE_SCALE_IMAGE)    
    faces = []
#         
    for (x,y,w,h) in faceRects:
        face = show_image[y:y + h, x:x + w]
        cv2.rectangle(show_image,(x,y),(x+w,y+h),(0,255,0),2)
        
        
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
    #     print(len(match_1[0]))
    #     print(len(match_1[1]))
        
        count = []   #dem so khuon mat co khoang cach Euclid < threshold cua tung nguoi trong dataSet
        for i in range(len(match_1)):
            tmpp = []
            for j in range(len(match_1[i])):
                dem = 0
    #             print(match_1[i][j])
                for k in range(len(match_1[i][j])):
                    if(match_1[i][j][k] < threshold):
                        dem = dem + 1
                tmpp.append(dem)
            count.append(np.asarray(tmpp))
#         print(count)
        
        
        
        people = []
        phantram = []
        for i in range(test_embs.shape[0]):
            min_distances = []
            for j in range(len(match_1[i])):
                min_distances.append(np.min(match_1[i][j]))
            
#             print(min_distances)
#             print(np.min(min_distances))
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
                    
        names = ['adele','chuan','taylor swift','ed sheeran','adam levine','khanh']
        names_1 = []
        print(phantram)
        #     title = ""
        for p in people:
            if p == "unknown":
                name = "unknown"
            else:
                name = names[p]
            names_1.append(name)
        #         title = title + name + " "
#                 
        for i,(x,y,w,h) in enumerate(faceRects):
            
            
            ten = names_1[i]
            print(ten)
            
            
            cv2.putText(show_image,names_1[i],(x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
            if(phantram[i]>0):
                cv2.putText(show_image,str(phantram[i]),(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),1)
                

#         show_image = imutils.resize(show_image,width = 720)   
    cv2.imshow("result",show_image)
            
     
    # cv2.imshow("result",show_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xff == ord("q"):
        exit()
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
            


