from django.shortcuts import render
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime
import base64

def button(request):
    return render(request, 'index.html')

def output(request):
    # Create the webcam object
    camera = cv2.VideoCapture(0)  # 0 represents the default webcam device

    # Check if the webcam is opened successfully
    if not camera.isOpened():
        output_data = "Failed to open webcam."
        return render(request, 'index.html', {'output_data': output_data})

    # Capture a single frame from the webcam
    ret, frame = camera.read()

    # Check if the frame is read successfully
    if not ret:
        output_data = "Failed to read frame from webcam."
        return render(request, 'index.html', {'output_data': output_data})

    # Convert the captured frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    # Initialize a counter for the red shapes
    red_shapes_count = 0

    # Iterate over the contours and filter red triangles
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon has 3 or more vertices, it is a valid shape
        if len(approx) >= 3:
            # Draw a bounding box around the triangle
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

            # Increment the red shapes count
            red_shapes_count += 1

    # Save the frame as an image if there are red-colored shapes detected
    if red_shapes_count > 0:
        # Generate a unique filename for each captured image
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join('captured_images', f"image_{red_shapes_count}_{current_time}.jpg")

        # Save the frame as an image
        cv2.imwrite(filename, frame)

        # Convert the image to base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the output data
        output_data = f"Number of red shapes: {red_shapes_count}"

        # Release the webcam
        camera.release()

        # Render the output template with the data
        return render(request, 'index.html', {'output_data': output_data, 'image_base64': image_base64})

    # Release the webcam
    camera.release()

    # Prepare the output data
    output_data = "No red shapes detected."

    # Render the output template with the data
    return render(request, 'index.html', {'output_data': output_data})
