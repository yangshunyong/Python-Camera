import cv2
import numpy as np
from deepface import DeepFace
import os

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]

FACE_DETECTION_OPENCV = True

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
    ret, frame = cap.read()
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

    try:
        verified = DeepFace.verify(frame, "ldh/1.jpg", model_name=models[1], detector_backend = detectors[0])
        result = verified.get("verified")

    except:
        result = False
        print("exception!");

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



