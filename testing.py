import cv2
import numpy as np

# Step 1: Image Acquisition
image = cv2.imread(r"C:\Users\lenovo\Desktop\aon\aonimages\alingot.jpg")

# Step 2: Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Thresholding and Segmentation
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, (3, 3))

# Step 4: Connected Component Analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)

# Step 5: Filtering and Counting
min_ingot_area = 1000  # Minimum area to consider as an ingot
count = 0

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_ingot_area:
        count += 1
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Step 6: Visualization or Output
cv2.putText(image, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
