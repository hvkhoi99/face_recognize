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


# INITIALIZE MODELS
nn4_small2 = create_model()

nn4_small2.summary()

nn4_small2.load_weights('weights/nn4.small2.v1.h5')
#shape_predictor_68_face_landmarks.dat: dc su dung de tim dac trung khuon mat
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#LOAD TRAINING INFORMATION
train_paths = glob.glob("image/*")
# train_paths = glob.glob("result/*")
# print(train_paths)

nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['image', 'label', 'name'])

for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[image,i,name]
        
# print(df_train)

# PRE-PROCESSING
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
    

# TRAINING
# label2idx = []
# 
# for i in tqdm(range(len(train_paths))):
#     label2idx.append(np.asarray(df_train[df_train.label == i].index))

train_embs = calc_embs(df_train.image)
np.save("train_embs.npy", train_embs)

    


# train_embs = np.concatenate(train_embs)
# 
# # ANALYSING
# import matplotlib.pyplot as plt
# 
# match_distances = []
# for i in range(nb_classes):
#     ids = label2idx[i]
#     distances = []
#     for j in range(len(ids) - 1):
#         for k in range(j + 1, len(ids)):
#             distances.append(distance.euclidean(train_embs[ids[j]].reshape(-1), train_embs[ids[k]].reshape(-1)))
#     match_distances.extend(distances)
#     
# unmatch_distances = []
# for i in range(nb_classes):
#     ids = label2idx[i]
#     distances = []
#     for j in range(20):
#         idx = np.random.randint(train_embs.shape[0])
#         while idx in label2idx[i]:
#             idx = np.random.randint(train_embs.shape[0])
#         distances.append(distance.euclidean(train_embs[ids[np.random.randint(len(ids))]].reshape(-1), train_embs[idx].reshape(-1)))
#     unmatch_distances.extend(distances)
#     
# _,_,_=plt.hist(match_distances,bins=100)
# _,_,_=plt.hist(unmatch_distances,bins=100,fc=(1, 0, 0, 0.5))
# 
# plt.show()