import cv2
import numpy as np
import os
from datetime import datetime

# IP camera configuration
ip_address = '192.168.1.64'  # Replace with the IP address of your camera
username = 'admin'  # Replace with your camera's username
password = 'password@123'  # Replace with your camera's password
rtsp_port = '554'  # Replace with the RTSP port of your camera (default: 554)

# Create the RTSP URL for the IP camera stream
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{rtsp_port}/live"

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

image_bw = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
image_color = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)

# Rest of the code for red triangle detection
# Convert the captured frame to HSV color space
hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the red color range
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Threshold the image to extract the red regions
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a counter for the red triangles
red_triangle_count = 0

# Iterate over the contours and filter red triangles
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the polygon has 3 vertices, it is a triangle
    if len(approx) >= 3:
        # Draw a bounding box around the triangle
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

        # Increment the red triangle count
        red_triangle_count += 1

# Display the image with detected triangles
cv2.imshow("Red Triangles Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the count of red triangles
print("Number of red triangles:", red_triangle_count)

# Save the frame as an image if there are red triangles detected
if red_triangle_count > 0:
    # Generate a unique filename for each captured image
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(save_folder, f"image_{red_triangle_count}_{current_time}.jpg")

    # Save the frame as an image
    cv2.imwrite(filename, frame)
