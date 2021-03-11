
import cv2
from imageio import imread
from keras.models import load_model
import dlib
from model import create_model
from align import AlignDlib



# INITIALIZE MODELS
# nn4_small2 = create_model()

# nn4_small2.summary()
# 
# nn4_small2.load_weights('weights/nn4.small2.v1.h5')

# alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')
