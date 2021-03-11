import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import distance
import dlib
import cv2
from model import create_model
from align import AlignDlib
import RPi.GPIO as GPIO
import time
import os

# Setup the GPIO PIN
GPIO.setmode(GPIO.BOARD)
GPIO.setup(22, GPIO.OUT)
p= GPIO.PWM(22, 50)
p.start(13)

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



# TRAINING
label2idx = []
train_embs = np.load("train_embs.npy",allow_pickle=True)
train_embs = np.concatenate(train_embs)

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))
#     print(df_train[df_train.label == i])
#     print(np.asarray(df_train[df_train.label == i].index))
#     print(label2idx)

match_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
#     print(ids)
    distances = []
    for j in range(len(ids) - 1):
        for k in range(j + 1, len(ids)):
            distances.append(distance.euclidean(train_embs[ids[j]].reshape(-1), train_embs[ids[k]].reshape(-1)))
    match_distances.extend(distances)
    
unmatch_distances = []
for i in range(nb_classes):
    ids = label2idx[i]
    distances = []
    for j in range(10):
        idx = np.random.randint(train_embs.shape[0])
        while idx in label2idx[i]:
            idx = np.random.randint(train_embs.shape[0])
        distances.append(distance.euclidean(train_embs[ids[np.random.randint(len(ids))]].reshape(-1), train_embs[idx].reshape(-1)))
    unmatch_distances.extend(distances)
# print(len(unmatch_distances))

_,_,_=plt.hist(match_distances,bins=130)
_,_,_=plt.hist(unmatch_distances,bins=130,fc=(1, 0, 0, 0.5))
# plt.show()
#
threshold = 0.75

test_image = cv2.imread("test_image/asdx.jpg")
show_image = test_image.copy()

hogFaceDetector = dlib.get_frontal_face_detector()
faceRects = hogFaceDetector(test_image, 0)

faces = []
    
for faceRect in faceRects:
    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()
    face = test_image[y1:y2,x1:x2]
        
    faces.append(face)
    
    cv2.rectangle(show_image,(x1,y1),(x2,y2),(0,255,0),2)
print("len(faces) = {0}".format(len(faceRects)))

if(len(faces)==0):
    print("no face detected!")

if(len(faces)>0):  
    test_embs = calc_emb_test(faces)
    test_embs = np.concatenate(test_embs)
    print(test_embs)
    people = []
    for i in range(test_embs.shape[0]):
        distances = []
        for j in range(len(train_paths)):
            distances.append(np.min([distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
        print(distances)
        print(np.min(distances))
        if np.min(distances)>threshold:
            people.append("unknown")
        else:
            p.ChangeDutyCycle(12)
            time.sleep(3)
            p.ChangeDutyCycle(2)
            time.sleep(1)
            p.ChangeDutyCycle(0)
            for a in range(len(distances)):
                if(distances[a] == np.min(distances)):
                    res = a
                    print(res)
                    people.append(res)
                    
            
#     names = ['adele','chuan','taylor swift','ed sheeran','adam levine','khanh']
    names = os.listdir("image/")
    names_1 = []
    
    for p in people:
        if p == "unknown":
            name = "unknown"
        else:
            name = names[p]
        names_1.append(name)
#         print(name)
              
    for i,faceRect in enumerate(faceRects):
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        
        
        cv2.putText(show_image,names_1[i],(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        

cv2.imshow("image", show_image)
cv2.waitKey(0)



    