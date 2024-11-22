# Description: This program is to recognize faces using deepface library with threading
# The python version is 3.10.10. There may be some problem if the python version is not
# 3.10.10.

import cv2
import numpy as np
from deepface import DeepFace
import os
import threading

# Most models take about 0.45s except mtcnn takes about 1.64s
# Facenet & mtcnn are the most accurate models, but take longer time
# Facenet & ssd is the most balanced combination
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
model_name = models[1]
detectors_backend = detectors[1]

# Do recognition every RECOGNIZE_FRAME frame
RECOGNIZE_FRAME = 60

# Initialize the model to avoid first frame delay
DeepFace.verify("ldh/1.JPG", "ldh/2.JPG", model_name=model_name, detector_backend = detectors_backend)

def recognize_faces(frame):
    global result

    try:
        # Definition a variable to save current time
        start_time = cv2.getTickCount()     
        verified = DeepFace.verify(frame_copy, "ldh/1.jpg", model_name = model_name, detector_backend = detectors_backend)
        result = verified.get("verified")
        print(verified)
        print("Distance" + str(verified.get("distance")));
        print("Threshold:"+ str(verified.get("threshold")))
        print("Time:"+ str(verified.get("time")))
        # Calculate time elapsed in seconds
        time_elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        print("Time elapsed: {:.2f} seconds".format(time_elapsed))
    except:
        result = False
        print("exception!");

    print("------------:" + str(result))

# Init the camera 0
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#define a variable to store the frame number
frame_number = 0
#define a variable to store the frame rate
frame_rate = 30.0
#define a variable to store current time in seconds
start_time = cv2.getTickCount()

result = False

# Read frames and show untile user press 'q' to quit
while True:
    ret, orig_frame = cap.read()
    frame = cv2.flip(orig_frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
# increase frame number by 1
    frame_number += 1

# Calculate time elspased in seconds from start_time
    time_in_seconds = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

# Calculate frame rate by frame_number and time_in_seconds
    frame_rate = frame_number / time_in_seconds
# Set frame_rate to 0 if time_in_seconds less than 2 seconds
    if time_in_seconds < 2:
        frame_rate = 0.0

# overlay frame number with the frame
    cv2.putText(frame, 'Frame Number: {}'.format(frame_number), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# overlay frame rate with the frame
# Truncate frame_rate to keep only 2 decimal places
    frame_rate = "{:.2f}".format(frame_rate)
    cv2.putText(frame, 'Frame Rate: {}'.format(frame_rate) + 'fps', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    

    if (frame_number % RECOGNIZE_FRAME == 0):
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        thread = threading.Thread(target=recognize_faces, args=(frame_copy,))
        thread.start()

    if (result):
        cv2.putText(frame, "YES", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) 
    else:
        cv2.putText(frame, "NO", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
# show the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if (cv2.waitKey(1) == ord('q')) or (cv2.waitKey(1) == ord('Q')):
        break

# release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()



