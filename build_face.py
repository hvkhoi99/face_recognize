from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os

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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_id = input("\n Enter user id :") 
print ("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # convert frame to array
    image = frame.array
    image_1  = frame.array
    #Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #Look for faces in the image using the loaded cascade file
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    print ("Found "+str(len(faces))+" face(s)")
    #Draw a rectangle around every found face
    for (x,y,w,h) in faces:
        image_1 = image_1[y :y + h , x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#         print(x,y,w,h)
    #Save the result image
    if len(faces):
        count = count + 1
        img_item = path + "/"+ str(count) + ".jpg"
        cv2.imwrite(img_item,image_1)
    # display a frame    
    cv2.imshow("Frame", image)
    #wait for 'q' key was pressed and break from the loop
    if cv2.waitKey(1) & 0xff == ord("q"):
        exit()
    if count == 20:
        exit()
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
