import cv2
import numpy as np
import os
import pyodbc
from datetime import datetime
import subprocess
import sys

def is_camera_pinging(ip_address):
    # Execute the ping command
    process = subprocess.Popen(['ping', '-n', '1', '-w', '1000', ip_address], stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Check the output for a successful ping
    if "Reply from {}".format(ip_address) in output.decode():
        return True
    else:
        return False

def read_camera_credentials(camera_id):
    # Connect to the SQL Server database
    cnxn = pyodbc.connect("Driver={SQL Server};"
                          "Server=LAPTOP-C3GTL6B2\SQLEXPRESS;"
                          "Database=INGOT_Counting;"
                          "Trusted_Connection=yes;")
    try:
        # Create a cursor object to execute SQL queries
        cursor = cnxn.cursor()

        # Fetch the camera credentials from the database for the given camera ID
        cursor.execute("SELECT ip_address, username, password, rtsp_port FROM camera_credentials WHERE camera_id = ?", camera_id)
        row = cursor.fetchone()

        # Check if the row is not empty
        if row:
            ip_address, username, password, rtsp_port = row
            return {
                'ip_address': ip_address,
                'username': username,
                'password': password,
                'rtsp_port': rtsp_port
            }
        else:
            print("Camera credentials not found for camera ID:", camera_id)
            return None
    except Exception as e:
        print("Error occurred while fetching camera credentials from the database:", e)
        return None
    finally:
        # Close the database connection
        cnxn.close()

# Get the directory path of the current script
script_dir = sys.path[0]

# Specify the path to the text file containing camera IDs
camera_ids_file = os.path.join(script_dir, "camera_credentials.txt")

# Read camera IDs from the text file
with open(camera_ids_file, 'r') as f:
    camera_ids = f.read().splitlines()

# Iterate over camera IDs
for camera_id in camera_ids:
    # Read camera credentials from the database for the current camera ID
    camera_credentials = read_camera_credentials(camera_id)

    # Check if camera credentials are fetched successfully
    if camera_credentials:
        # Extract the credentials
        ip_address = camera_credentials.get('ip_address')
        username = camera_credentials.get('username')
        password = camera_credentials.get('password')
        rtsp_port = camera_credentials.get('rtsp_port')

        # Check if the necessary credentials are present
        if not all([ip_address, username, password, rtsp_port]):
            print("Incomplete camera credentials. Please ensure all fields are populated for camera ID:", camera_id)
            continue

        # Check if the camera is pinging
        if is_camera_pinging(ip_address):
            print("Camera", camera_id, "is pinging.")
        else:
            print("Camera", camera_id, "is not pinging.")
            continue

        # Create the RTSP URL for the IP camera stream
        rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{rtsp_port}/live"

        # Define the lower and upper bounds for the red color range
        lower_red = np.array([150, 50, 20])
        upper_red = np.array([179, 255, 255])

        # Connect to the IP camera stream
        camera = cv2.VideoCapture(rtsp_url)

        # Check if the camera stream is opened successfully
        if not camera.isOpened():
            print("Failed to open IP camera stream for camera ID:", camera_id)
            continue

        # Create a folder to store the captured images
        save_folder = 'E://Captured_Images//'

        # Capture a single frame from the camera stream
        ret, frame = camera.read()

        # Release the camera stream
        camera.release()

        # Check if the frame is read successfully
        if not ret:
            print("Failed to read frame from IP camera for camera ID:", camera_id)
            continue

        # Define the rectangular mask coordinates (top-left and bottom-right points)
        mask_top_left = (600, 200)  # (x, y) coordinates of the top-left point
        mask_bottom_right = (1200, 1000)  # (x, y) coordinates of the bottom-right point

        # Create an empty mask image of the same size as the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Draw a filled white rectangle on the mask
        cv2.rectangle(mask, mask_top_left, mask_bottom_right, (255, 255, 255), cv2.FILLED)

        # Apply the mask to the frame and thresholded mask
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
        hsv_masked = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
        mask_masked = cv2.inRange(hsv_masked, lower_red, upper_red)

        # Apply morphological operations (erosion and dilation) to refine the shape boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_masked = cv2.morphologyEx(mask_masked, cv2.MORPH_OPEN, kernel)

        # Find contours in the masked thresholded image
        contours, _ = cv2.findContours(mask_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a counter for the red shapes
        red_shapes_count = 0

        # Iterate over the contours and filter red shapes based on size
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the contour has exactly 3 vertices and its area is within the desired range, it is a triangle
            contour_area = cv2.contourArea(contour)
            if len(approx) >= 3 and 100 < contour_area:
                # Draw a bounding box around the triangle
                cv2.drawContours(frame_masked, [approx], 0, (0, 255, 0), 2)

                # Increment the red shapes count
                red_shapes_count += 1

        # Display the masked image with detected shapes
        cv2.imshow("Red shapes Detection", frame_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print the count of red shapes
        print("Number of red shapes for camera ID", camera_id + ":", red_shapes_count)

        # Save the frame as an image if there are red shapes detected
        if red_shapes_count > 0:
            # Generate a unique filename for each captured image
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = (save_folder + f"image_{red_shapes_count}_{current_time}.jpg")
            # print((filename))

            # Save the masked frame as an image
            cv2.imwrite(filename, frame_masked)

            # Connect to the SQL Server database
            cnxn = pyodbc.connect("Driver={SQL Server};"
                                  "Server=LAPTOP-C3GTL6B2\SQLEXPRESS;"
                                  "Database=INGOT_Counting;"
                                  "Trusted_Connection=yes;")

            try:
                # Create a cursor object to execute SQL queries
                cursor = cnxn.cursor()

                # Insert the data counts and image paths into the database
                cursor.execute("EXEC SaveImageData ?, ?, ?, ?", red_shapes_count, red_shapes_count, filename, camera_id)

                # Commit the transaction to save the changes
                cnxn.commit()

                print("Data successfully saved in the database for camera ID:", camera_id)
            except Exception as e:
                print("Error occurred while saving data in the database for camera ID:", camera_id, "Error:", e)
            finally:
                # Close the database connection
                cnxn.close()