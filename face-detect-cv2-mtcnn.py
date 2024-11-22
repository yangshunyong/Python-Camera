import cv2
from mtcnn.mtcnn import MTCNN

FACE_DETECTION_OPENCV = False

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

# Read frames and show untile user press 'q' to quit
while True:
    ret, orig_frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
   
    frame = cv2.flip(orig_frame, 1)
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

# scale frame to match window szie
    frame = cv2.resize(frame, (640, 480))

# Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Convert the frame to Yuyv and save to a file in binary mode
    yuyv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_YV12 )
# Save yuyv to a file in binary mode
    with open('yuyv.bin', 'wb') as f:
        f.write(yuyv)

    if (FACE_DETECTION_OPENCV) :
    # Load the Haar cascade face detection with configuration file "haarcascade_frontalface_default.xml"
    # haarcascade_frontalface_default.xml for face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face by cascade in the gray image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw rectangle around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        # Create an instance of the MTCNN detector
        detector = MTCNN()
        # Convert fream from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = detector.detect_faces(rgb_frame)
        # Draw rectangle around the face
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
      
    
# show the frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if (cv2.waitKey(1) == ord('q')) or (cv2.waitKey(1) == ord('Q')):
        break

# release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()



