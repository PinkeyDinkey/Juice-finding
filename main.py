import numpy as np
import pyautogui
import win32api, win32con, win32gui
import cv2
import math
import os
import time
import keyboard
CONFIG_FILE = './yolov4-obj.cfg'
WEIGHT_FILE = './yolov4-o.weights'


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHT_FILE)
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

path_name = "Data/train_0.jpg"
image = cv2.imread(path_name)
# создать 4D blob
image = np.array(image)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
# получаем имена всех слоев
# прямая связь (вывод) и получение выхода сети
# измерение времени для обработки в секундах
layerOutputs = net.forward(ln)

boxes = []
confidences = []
frame_height, frame_width = image.shape[:2]
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence <1:
            box = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
print(confidence)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.6)
if len(indices) > 0:
    print(f"Detected:{len(indices)}")
    min = 99999
    min_at = 0
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
resized = cv2.resize(image, (600,600), interpolation = cv2.INTER_AREA)
cv2.imshow("image", resized)
cv2.waitKey(0)



