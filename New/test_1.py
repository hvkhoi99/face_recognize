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
label2idx = []
train_embs = np.load("train_embs.npy",allow_pickle=True)
train_embs = np.concatenate(train_embs)

for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))
#     print(df_train[df_train.label == i])
#     print(np.asarray(df_train[df_train.label == i].index))
# print(label2idx[5])

match_distances = []
for i in range(nb_classes): #nb_class: so luong nguoi trong dataSet
    ids = label2idx[i]      #
    distances = []
    for j in range(len(ids) - 1):
        for k in range(j + 1, len(ids)):
            distances.append(calc_distance_euclid(train_embs[ids[j]], train_embs[ids[k]]))
    match_distances.extend(distances)
# print(distances[0])
    
#     
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
# print(len(unmatch_distances))
# 
_,_,_=plt.hist(match_distances,bins=130)
_,_,_=plt.hist(unmatch_distances,bins=130,fc=(1, 0, 0, 0.5))
# plt.show()

threshold = 0.75


test_image = cv2.imread("test_image/Phan Minh Chuan_8.jpg")
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
    
print("len(faces) = {0}".format(len(faceRects)))

if(len(faces)==0):
    print("no face detected!")

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
    print(count)
    
    
    
    people = []
    phantram = []
    for i in range(test_embs.shape[0]):
        min_distances = []
        for j in range(len(match_1[i])):
            min_distances.append(np.min(match_1[i][j]))
        
        print(min_distances)
        print(np.min(min_distances))
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
            name = "unknown"
        else:
            name = names[p]
        names_1.append(name)

              
    for i,faceRect in enumerate(faceRects):
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        print(names_1[i])
        
        cv2.rectangle(show_image,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(show_image,names_1[i],(x1,y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        if(phantram[i]>0):
            cv2.putText(show_image,str(phantram[i]),(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        

cv2.imshow("image", show_image)
cv2.waitKey(0)



    
