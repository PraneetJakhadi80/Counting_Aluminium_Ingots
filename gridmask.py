import cv2
import numpy as np

# Function to draw grid on frame
def draw_grid(frame, grid_rows, grid_cols, mask_top_left, mask_width, mask_height, line_thickness):
    cell_width = mask_width // grid_cols
    cell_height = mask_height // grid_rows
    grid_color = (255, 255, 255)  # Red color (BGR format)

    # Draw vertical grid lines
    for i in range(1, grid_cols):
        x = mask_top_left[0] + i * cell_width
        cv2.line(frame, (x, mask_top_left[1]), (x, mask_top_left[1] + mask_height), grid_color, line_thickness)

    # Draw horizontal grid lines
    for i in range(1, grid_rows):
        y = mask_top_left[1] + i * cell_height
        cv2.line(frame, (mask_top_left[0], y), (mask_top_left[0] + mask_width, y), grid_color, line_thickness)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define mask dimensions
mask_top_left = (200, 100)
mask_width = 400
mask_height = 300

# Define background color
background_color = (255, 255, 255)  # White color (BGR format)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # Create a blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, mask_top_left, (mask_top_left[0] + mask_width, mask_top_left[1] + mask_height), 255, -1)

    # Apply mask to frame
    masked_frame = cv2.bitwise_or(frame, cv2.cvtColor(cv2.bitwise_not(mask), cv2.COLOR_GRAY2BGR), mask=mask)
    masked_frame[np.where((masked_frame == [0, 0, 2]).all(axis=2))] = background_color

    # Draw grid on the masked frame
    grid_rows = 8
    grid_cols = 8
    line_thickness = 5
    draw_grid(masked_frame, grid_rows, grid_cols, mask_top_left, mask_width, mask_height, line_thickness)

    # Convert masked frame to grayscale
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Define the region of interest within the grid
    cell_width = mask_width // grid_cols
    cell_height = mask_height // grid_rows
    roi_top_left = (mask_top_left[0] + cell_width, mask_top_left[1] + cell_height)
    roi_bottom_right = (mask_top_left[0] + cell_width * 7, mask_top_left[1] + cell_height * 7)

    # Extract the region of interest
    roi_frame = gray_frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Apply threshold to detect black shapes within the ROI
    _, thresholded = cv2.threshold(roi_frame, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and draw bounding rectangles
    black_shapes = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold for detection
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(masked_frame, (x + roi_top_left[0], y + roi_top_left[1]),
                          (x + roi_top_left[0] + w, y + roi_top_left[1] + h), (0, 255, 0), 2)
            black_shapes += 1

    # Display the number of black shapes
    cv2.putText(masked_frame, f"Black Shapes: {black_shapes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the masked frame
    cv2.imshow("Masked Frame with Grid", masked_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
