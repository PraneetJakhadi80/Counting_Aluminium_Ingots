import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Load the image
image = cv2.imread(r"C:\Users\lenovo\Desktop\aon\aonimages\sss.jpg")
# image = cv2.imread(r"C:\Users\lenovo\Desktop\aon\aonimages\square-steel-pipe.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a window to hold trackbars
cv2.namedWindow("parameters")
cv2.resizeWindow("parameters", 640, 240)
cv2.createTrackbar("Threshold1", "parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2", "parameters", 255, 255, empty)

while True:
    imgblur = cv2.GaussianBlur(gray, (7, 7), 1)

    threshold1 = cv2.getTrackbarPos("Threshold1", "parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "parameters")
    imgcanny = cv2.Canny(imgblur, threshold1, threshold2)

    # Find contours in the image
    contours, _ = cv2.findContours(imgcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of square/rectangle rings
    num_rings = 0

    # Loop over the contours
    for contour in contours:
        # Approximate the contour to simplify its shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 or more vertices (square or rectangle shape)
        # contour_area = cv2.contourArea(contour)
        if len(approx)>= 4 and len(approx)<6 :
            # Calculate the aspect ratio of the bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Set an aspect ratio threshold to identify square/rectangle rings
            aspect_ratio_threshold = 0.8  # Adjust this value as needed

            # Draw the ring and increment the ring count if the aspect ratio is above the threshold
            if aspect_ratio >= aspect_ratio_threshold:
                cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)
                num_rings += 1

    # Display the ring count on the image
    cv2.putText(image, "Ring Count: " + str(num_rings), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    imagestack = stackImages(0.8, ([image, imgcanny]))

    cv2.imshow("Result", imagestack)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
