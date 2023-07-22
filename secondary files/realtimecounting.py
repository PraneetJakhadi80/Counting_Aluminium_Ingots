import cv2
import numpy as np

# Connect to the webcam
camera = cv2.VideoCapture(1)

# Set camera resolution (adjust as needed)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Capture a single image from the camera
ret, frame = camera.read()

# Release the camera
camera.release()

# Convert the captured frame to HSV color space


# Define the lower and upper bounds for the red color range
lower_red = np.array([160, 150, 20])
upper_red = np.array([180, 255, 255])

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Threshold the image to extract the red regions
mask = cv2.inRange(hsv, lower_red, upper_red)
# mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
# mask = cv2.bitwise_or(mask1, mask2)
cv2.imshow("mask", mask)

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
