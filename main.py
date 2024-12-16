from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import serial
import threading
import numpy as np
import csv 
import folium
import pandas as pd


# Global variables for coordinates
global latitude, longitude
latitude, longitude = None, None
lock = threading.Lock()

csv_filename = 'new5vidercoordinates.csv'


csv_file = open(csv_filename, 'a', newline='')  # Open the file in append mode
csv_writer = csv.writer(csv_file)

# If file doesn't exist or is empty, write headers
#if not file_exists:
headers = ["X1", "Y1","X2", "Y2", "ID", "Classname", "Latitude", "Longitude", "conf"]

csv_writer.writerow(headers)  # Write headers'''
# csv modification 


def coordinate():
    global latitude, longitude
    ser = serial.Serial('COM5', 9600)  # Open serial port
    while True:
        data = ser.readline().decode('utf-8').strip()  # Read and decode data
        if data.startswith("Latitude="):
            with lock:
                latitude = float(data.split("=")[1].split()[0])  # Extract latitude
                longitude = float(data.split("Longitude=")[1])  # Extract longitude

# Start GPS reading in a separate thread
thread = threading.Thread(target=coordinate)
thread.daemon = True  # Daemonize thread
thread.start()

# Load model and other initializations
model = YOLO(r"C:\Users\Sony\Desktop\300 epochs\weights\best.pt")
className = ["alligator", "edge", "longitudinal", "patching", "potholes", "rutting", "traverse"]
#mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.4)
#camera = "https://192.168.100.5:8080/video"
#video = cv2.VideoCapture(camera)
#video.open(camera)
#video = cv2.VideoCapture(r"C:\Users\Hp\Downloads\New folder (4)\newvideo.mp4")
video = cv2.VideoCapture(0)
video.set(3, 480)
video.set(4, 640)
limits = [640, 310, 0, 310]
totalCount = [1,2]
final = []
unique_results = {}

fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # *"XVID"

output = cv2.VideoWriter("output3.avi",fourcc,7.0,(640,360))

mask_color = (0,0,0)
mask = np.full((360, 640, 3), mask_color, dtype=np.uint8)
mask = cv2.rectangle(mask, (0, 150), (640, 360), (255, 255, 255), -1)

while True:
    ret, frame = video.read()
    frame=cv2.resize(frame,(640,360))
    imgRegion = cv2.bitwise_and(frame, mask)
    if not ret:
        break
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = className[cls]
            if currentClass in ["alligator", "edge", "longitudinal", "patching", "potholes", "rutting", "traverse"] and conf > 0.4:
                with lock:
                    if latitude is not None and longitude is not None:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (50, 100, 210), 5)
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        with lock:
            cx, cy = x1 + w // 2, y1 + h // 2
            if limits[2] < cx < limits[0] and limits[1] - 20 < cy < limits[1] + 20: 
                if latitude is not None and longitude is not None:
                    result = np.append(result, [currentClass, latitude, longitude,conf])
                 
                    csv_writer.writerow(result[0:9])
                    print(result[0:9])

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, colorR=(225, 0, 0))
        cvzone.putTextRect(frame, f'{className[cls]} {Id} {conf}' , (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255, 0, 0), offset=3)

  

   # cvzone.putTextRect(frame, f'holes: {len(totalCount)}', (50, 50), offset=3, thickness=1, scale=1, colorR=(50, 10, 200))
    cv2.imshow("phone camera video", frame)
    output.write(frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break
csv_file.close()  # Close the CSV file when done
output.release()
cv2.destroyAllWindows()