from ultralytics import YOLO
import cv2 as cv
import cvzone 
import math

# For an image
"""
image = cv.imread("rx7.jpg")


model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model(source=image, show=True)
cv.waitKey(0)
"""


# Trying to run it with webcam
model = YOLO('../Yolo-Weights/yolov8n.pt')


classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


cap = cv.VideoCapture("Video_Path_here")
cap.set(3, 1280)
cap.set(4, 720)
while True:
    res, frame = cap.read()
    results = model(source=frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]   ## Get the bounding box 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            probability = box.conf[0]
            probability = (math.ceil(probability*100))/100    
            cls = int(box.cls[0])
            if classes[cls] == 'car':
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f'{classes[cls]} {probability}', (x1, y1-30))

    cv.imshow('Video', frame)
    if cv.waitKey(1) and 0xFF == ord('f'):
        break

cap.release()
cv.destroyAllWindows()