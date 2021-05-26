
# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
import os
from keras.preprocessing import image
model = load_model('facefeatures_new_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)

values = os.listdir('custom-face/train/')

print(values)
# while True:
#     _, frame = video_capture.read()
#     #canvas = detect(gray, frame)
#     #image, face =face_detector(frame)
    
#     face=face_extractor(frame)
#     if type(face) is np.ndarray:
#         face = cv2.resize(face, (224, 224))
#         im = Image.fromarray(face, 'RGB')
#            #Resizing into 128x128 because we trained the model with this image size.
#         img_array = np.array(im)
#                     #Our keras model used a 4D tensor, (images x height x width x channel)
#                     #So changing dimension 128x128x3 into 1x128x128x3 
#         img_array = np.expand_dims(img_array, axis=0)
#         pred = model.predict(img_array)
#         print(pred)

#         max = int(np.argmax(pred, axis=1))
#         print(max)
#         print(pred[0][max])
#         name="None matching"
        
#         if(pred[0][max] >0.5):
#             name= str(values[max])
#         cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#     else:
#         cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()

for x in os.listdir('./custom-face/test/Elroi/'):
    image = './custom-face/test/Abeni/' + str(x)
    print(image)
    img = cv2.imread(image)
    # cv2.imshow("I want see you",img)
    # cv2.waitKey(0)
    print(type(img))
    img = cv2.resize(img, (224,224))
    im = Image.fromarray(img, 'RGB')
    img_array = np.array(im)
    # x = face_extractor(image)
    # face = cv2.resize(x, (224, 224))
    # im = Image.fromarray(face, 'RGB')
    # # im = './custom-face/test/81.jpg'
    # img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    print(pred)
    max = int(np.argmax(pred, axis=1))
    print(max)
    print(pred[0][max])
    name="None matching"

    # if(pred[0][max] >0.5):
    name= str(values[max])
    print(name)