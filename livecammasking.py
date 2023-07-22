import cv2
import numpy as np

# IP camera configuration
ip_address = '192.168.1.64'  # IP address of your camera
username = 'admin'  # camera's username
password = 'password@123'  # camera's password
rtsp_port = '554'  # RTSP port of your camera (default: 554)

# Create the RTSP URL for the IP camera stream
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{rtsp_port}/live"

# Create a VideoCapture object to connect to the IP camera stream
camera = cv2.VideoCapture(rtsp_url)

# Check if the camera stream is opened successfully
if not camera.isOpened():
    print("Failed to open IP camera stream.")
    exit()

# Define the rectangular mask coordinates (top-left and bottom-right points)
mask_top_left = (600, 200)  # (x, y) coordinates of the top-left point
mask_bottom_right = (1200, 1000)  # (x, y) coordinates of the bottom-right point
# Process the camera stream
while True:
    # Read a frame from the camera stream
    ret, frame = camera.read()

    # Check if the frame is read successfully
    if not ret:
        print("Failed to read frame from IP camera.")
        break

    # Create an empty mask image of the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Draw a filled white rectangle on the mask
    cv2.rectangle(mask, mask_top_left, mask_bottom_right, (255, 255, 255), cv2.FILLED)

    # Apply the mask to the frame
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the masked frame
    cv2.imshow("Live Masking", frame_masked)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
