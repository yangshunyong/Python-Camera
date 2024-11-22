import cv2
import uvc

def list_uvc_cameras():
    devices = uvc.device_list()
    return devices

def print_camera_info(devices):
    for device in devices:
        print(device)

def list_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras
import uvc

if __name__ == "__main__":
    cameras = list_cameras()
    if cameras:
        print("Available cameras:")
        for cam in cameras:
            print(f"Camera {cam}")
    else:
        print("No cameras found.")

    devices = list_uvc_cameras()
    if devices:
        print("Available UVC cameras:")
        print_camera_info(devices)
    else:
        print("No UVC cameras found.")

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

    # scale frame to match window szie
        frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the frame to Yuyv and save to a file in binary mode
        yuyv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_YV12 )
    # Save yuyv to a file in binary mode
        with open('yuyv.bin', 'wb') as f:
            f.write(yuyv)
        
    # show the frame
        cv2.imshow('frame', frame)

        # Press 'q' to quit
        if (cv2.waitKey(1) == ord('q')) or (cv2.waitKey(1) == ord('Q')):
            break

    # release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()



