import cv2
import numpy as np
import os
from datetime import datetime

# IP camera configuration
ip_address = '192.168.1.64'  # IP address of your camera
username = 'admin'  # camera's username
password = 'password@123'  # camera's password
rtsp_port = '554'  # RTSP port of your camera (default: 554)

# Create the RTSP URL for the IP camera stream
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{rtsp_port}/live"

# Define the lower and upper bounds for the red color range
lower_red = np.array([150, 150, 20])
upper_red = np.array([179, 255, 255])

# Connect to the IP camera stream
camera = cv2.VideoCapture(rtsp_url)

# Check if the camera stream is opened successfully
if not camera.isOpened():
    print("Failed to open IP camera stream.")
    exit()

# Create a folder to store the captured images
save_folder = 'captured_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Capture a single frame from the camera stream
ret, frame = camera.read()

# Release the camera stream
camera.release()

# Check if the frame is read successfully
if not ret:
    print("Failed to read frame from IP camera.")
    exit()

# Convert the captured frame to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Threshold the image to extract the red regions
mask = cv2.inRange(hsv, lower_red, upper_red)

# Apply morphological operations (erosion and dilation) to refine the shape boundaries
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a counter for the red triangles
red_triangles_count = 0

# Iterate over the contours and filter red triangles based on size
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the contour has exactly 3 vertices and its area is within the desired range, it is a triangle
    contour_area = cv2.contourArea(contour)
    if len(approx) >= 3 and 1000 < contour_area :
        # Draw a bounding box around the triangle
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

        # Increment the red triangles count
        red_triangles_count += 1

# Display the image with detected triangles
cv2.imshow("Red Triangle Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the count of red triangles
print("Number of red triangles:", red_triangles_count)

# Save the frame as an image if there are red triangles detected
if red_triangles_count > 0:
    # Generate a unique filename for each captured image
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(save_folder, f"image_{red_triangles_count}_{current_time}.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
