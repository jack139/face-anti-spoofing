# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "" #  只使用 CPU

import sys

if len(sys.argv)<2:
    print("usage: python3 %s <image_path>" % sys.argv[0])
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import cv2
import time
#########################
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
#########################
from rPPG2.rPPG_Extracter import *
#from rPPG2.rPPG_lukas_Extracter import *
#########################
import face_recognition


# load YAML and create model
yaml_file = open("trained_model/RGB_rPPG_merge_softmax_.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
#model.load_weights("trained_model/RGB_rPPG_merge_softmax_.h5")
model.load_weights("batch_128_epochs_5_steps_100_0.h5")
print("[INFO] Model is loaded from disk")


dim = (128,128)
def get_rppg_pred(frame):
    use_classifier = False  # Toggles skin classifier
    sub_roi = []           # If instead of skin classifier, forhead estimation should be used set to [.35,.65,.05,.15]

    rPPG_extracter = rPPG_Extracter()
        
    rPPG = []

    rPPG_extracter.measure_rPPG(frame,use_classifier,sub_roi) 
    rPPG = np.transpose(rPPG_extracter.rPPG)

    return rPPG
    

def make_pred(li):
    [single_img,rppg] = li
    single_img = cv2.resize(single_img, dim)
    single_x = img_to_array(single_img)
    single_x = np.expand_dims(single_x, axis=0)
    single_pred = model.predict([single_x,rppg])
    return single_pred


    
cascPath = 'rPPG/util/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

collected_results = []
counter = 0          # count collected buffers
frames_buffer = 5    # how many frames to collect to check for
accepted_falses = 1  # how many should have zeros to say it is real

# Capture frame-by-frame
frame = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#faces = faceCascade.detectMultiScale(
#    gray,
#    scaleFactor=1.1,
#    minNeighbors=5
#)

faces = face_recognition.face_locations(gray)

#print(faces)

# Draw a rectangle around the faces
#for (x, y, w, h) in faces:
for (top, right, bottom, left) in faces:
    x, y, w, h = left, top, right-left+1, bottom-top+1
    sub_img=frame[y:y+h,x:x+w]
    #cv2.imwrite('img_%d_%d.jpg'%(x,y),sub_img)
    rppg_s = get_rppg_pred(sub_img)
    rppg_s = rppg_s.T

    pred = make_pred([sub_img,rppg_s])

    collected_results.append(np.argmax(pred))
    counter += 1

    print("Real: "+str(pred[0][0]))
    print("Fake: "+str(pred[0][1]))
    if len(collected_results) == frames_buffer:
        #print(sum(collected_results))
        if sum(collected_results) <= accepted_falses:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        collected_results.pop(0)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the resulting frame
#cv2.imwrite('img.jpg',frame)


# When everything is done, release the capture
cv2.destroyAllWindows()
