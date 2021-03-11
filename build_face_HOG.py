from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import dlib

id = input("Nhap id: ")
name = input("Nhap ten: ")
path = "image/"+id+"_"+name
if os.path.exists(path):
    print("Folder da ton tai")
else:
    print("Folder is created")
    os.mkdir(path)
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(320, 240))
#Load a cascade file for detecting faces
hogFaceDetector = dlib.get_frontal_face_detector()
# face_id = input("\n Enter user id :") 
print ("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # convert frame to array
    image = frame.array
    image_1  = frame.array
    #Convert to grayscale
    faceRects = hogFaceDetector(image, 0)
    #Look for faces in the image using the loaded cascade file

    print ("Found "+str(len(faceRects))+" face(s)")
    #Draw a rectangle around every found face
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        
        face = image[y1:y2,x1:x2]

        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
#         print(x,y,w,h)
    #Save the result image
    if len(faceRects):
        count = count + 1
        img_item = path + "/"+ str(count) + ".jpg"
        cv2.imwrite(img_item,face)
    # display a frame    
    cv2.imshow("Frame", image)
    #wait for 'q' key was pressed and break from the loop
    if cv2.waitKey(1) & 0xff == ord("q"):
        exit()
    if count == 20:
        exit()
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

