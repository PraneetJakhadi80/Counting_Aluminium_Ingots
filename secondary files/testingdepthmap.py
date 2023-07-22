import cv2
import numpy as np

# Path to the input image
image_path = r"C:\Users\lenovo\Desktop\aon\aonimages\alingot.jpg"

# Load the image
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur and Canny edge detection
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blurred, 30, 150)

# Dilate the edges
dilated = cv2.dilate(canny, (1, 1), iterations=2)

# Find contours
(contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Display the processed image with contours
cv2.imshow("Contour Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
